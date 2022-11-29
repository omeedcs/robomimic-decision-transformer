from argparse import ArgumentParser
import os
from functools import partial
from dataclasses import dataclass

import gym
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import wandb

from env_utils import make_robomimic_envs
from data_utils import torch_move_pad, load_robomimic_dset, robomimic_collate
from transformer import TransformerEncoder
from ff import FeedForwardEncoder
from agent import Agent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_parser():
    parser = ArgumentParser()
    p = parser.add_argument
    pr = partial(parser.add_argument, required=True)
    pb = partial(p, action="store_true")
    pr("--run_name")
    pr("--quality", choices=["ph", "mh", "mg", "all"])
    pr("--task", choices=["lift", "can", "square", "tool_hang"])
    pb("--log_to_wandb")
    p("--log_interval", type=int, default=250)
    p("--ckpt_interval", type=int, default=20)
    p("--log_dir", type=str, default="saves/")
    p("--dloader_workers", type=int, default=4)
    pb("--render")
    p("--warmup_steps", type=int, default=2000)
    p("--init_learning_rate", type=float, default=5e-4)
    p("--batch_size", type=int, default=256)
    p("--epochs", type=int, default=300)
    p("--l2_coeff", type=float, default=1e-4)
    p("--grad_clip", type=float, default=5.0)
    p("--eval_interval", type=int, default=50)
    pb("--argmax_actions_eval")
    p("--eval_timesteps", type=int, default=2000)
    p("--parallel_envs", type=int, default=8)
    p("--max_rollout_length", type=int, default=200)
    p(
        "--rtg_strategy",
        type=str,
        default="expert",
        choices=["percentiles", "expert"],
    )
    p(
        "--encoder_arch",
        type=str,
        default="transformer",
        choices=["transformer", "feedforward"],
    )
    p("--context_length", type=int, default=20)
    pb("--ignore_rtg")
    p("--d_model", type=int, default=200)
    p("--d_ff", type=int, default=800)
    p("--n_heads", type=int, default=6)
    p("--layers", type=int, default=4)
    p("--dropout_emb", type=float, default=0.05)
    p("--dropout_ff", type=float, default=0.05)
    p("--policy_dist", type=str, default="gmm", choices=["gmm", "gaussian"])
    p("--gmm_modes", type=int, default=5)

    p("--ckpt", type=int, default=None)
    pb("--eval")
    return parser


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        try:
            param = p.grad.data
        except AttributeError:
            continue
        else:
            param_norm = param.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


@dataclass
class Experiment:
    # Core
    run_name: str
    quality: str
    task: str
    log_to_wandb: bool = True
    hparams: dict = None

    # Misc.
    log_interval: int = 100
    ckpt_interval: int = 10
    log_dir: str = "saves/"
    dloader_workers: int = 6
    render: bool = False

    # Optimization
    warmup_steps: int = 2000
    init_learning_rate: float = 5e-4
    batch_size: int = 128
    epochs: int = 2
    l2_coeff: float = 1e-4
    grad_clip: float = 5.0

    # Eval
    async_envs: bool = True
    eval_interval: int = 200
    sample_during_eval: bool = True
    eval_timesteps: int = 2000
    parallel_envs: int = 10
    max_rollout_length: int = 200
    rtg_strategy: str = "percentiles"

    # Model
    encoder_arch: str = "transformer"
    context_length: int = 20
    ignore_rtg: bool = False
    d_model: int = 200
    d_ff: int = 800
    n_heads: int = 6
    layers: int = 4
    dropout_emb: float = 0.05
    dropout_ff: float = 0.05
    policy_dist: str = "gmm"
    gmm_modes: int = 5

    def print_key_hparams(self):
        print("\n\n\n--- Decision Transformer ---")
        print(f"task = {self.task}")
        print(f"dset quality = {self.quality}")
        print(f"encoder_arch = {self.encoder_arch}")
        print(f"context_length = {self.context_length}")
        print(f"ignore_rtg = {self.ignore_rtg}")
        print(f"rtg_strategy = {self.rtg_strategy}")
        print(f"policy_dist = {self.policy_dist}")
        print(f"parallel_envs = {self.parallel_envs}")
        print(f"env type = {type(self.envs)}")
        print("----------------------------\n\n\n")

    def start(self):
        assert self.quality in ["mh", "mg", "ph", "all"]
        assert self.task in ["can", "lift", "square", "tool_hang"]
        assert self.policy_dist in ["gmm", "gaussian"]
        assert self.encoder_arch in ["transformer", "feedforward"]
        assert self.rtg_strategy in ["percentiles", "expert"]
        self.init_envs_and_dsets()
        self.init_encoder()
        self.init_checkpoints()
        self.init_logger()
        self.print_key_hparams()

    def init_envs_and_dsets(self):
        # load dataset from disk using modified robomimic SequenceDataset
        dset_paths = []
        if self.quality == "all":
            qualities = ["mh", "ph", "mg"]
        else:
            qualities = [self.quality]
        for quality in qualities:
            path = os.path.join("datasets", self.task, quality, "dt.hdf5")
            if os.path.exists(path):
                dset_paths.append(path)
        assert (
            dset_paths
        ), "Dataset combination may be invalid or dt.hdf5 files may not have been generated"
        dsets = [
            load_robomimic_dset(path, context_length=self.context_length)
            for path in dset_paths
        ]
        self.dset = ConcatDataset(dsets)

        self.dloader = DataLoader(
            self.dset,
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            collate_fn=robomimic_collate,
            shuffle=True,
            pin_memory=True,
        )
        # create parallel envs, each with its own target return (which might be ignored by the agent)
        self.target_returns = self._select_target_returns()
        self.envs = make_robomimic_envs(
            dset_paths[0],
            context_length=self.context_length,
            target_returns=self.target_returns,
            render=self.render,
            parallel_envs=self.parallel_envs,
            async_envs=self.async_envs,
            max_rollout_length=self.max_rollout_length,
        )

    def _select_target_returns(self) -> np.ndarray:
        # NaN-based cheat to find valid returns in our dataset
        returns = []
        steps = []
        for batch in self.dloader:
            ret_batch = batch["demo_returns"]
            step_batch = batch["demo_steps"]
            for i in range(ret_batch.shape[0]):
                if not torch.isnan(ret_batch[i]):
                    returns.append(ret_batch[i].item())
                if not torch.isnan(step_batch[i]):
                    steps.append(step_batch[i].item())
        self.return_dist = np.array(returns)
        self.step_dist = np.array(steps)
        if self.rtg_strategy == "percentiles":
            # create targets that span the range of the return distribution
            qs = np.linspace(start=20, stop=100, num=self.parallel_envs)
            # (flipped so the rendered env is the highest return)
            rtgs = np.flip(np.percentile(self.return_dist, qs))
        elif self.rtg_strategy == "expert":
            q = np.percentile(self.return_dist, 95)
            rtgs = np.full((self.parallel_envs,), q)
        return rtgs

    def init_logger(self):
        if self.log_to_wandb:
            wandb.init(
                project="cs391r-decision-transformer",
                entity="cs391r-decision-transformer",
                dir="logs",
                name=self.run_name,
                config=self.hparams or {},
            )

    def init_encoder(self):
        action_dim = self.envs.action_space[0].shape[-1]
        obs_dim = self.envs.observation_space.shape[-1]
        rtg_dim = 1
        input_dim = obs_dim + action_dim + rtg_dim

        if self.encoder_arch == "transformer":
            encoder = TransformerEncoder(
                input_dim=input_dim,
                d_model=self.d_model,
                d_ff=self.d_ff,
                max_seq_len=self.context_length,
                n_heads=self.n_heads,
                layers=self.layers,
                dropout_emb=self.dropout_emb,
                dropout_ff=self.dropout_ff,
            )
        elif self.encoder_arch == "feedforward":
            encoder = FeedForwardEncoder(
                input_dim=input_dim,
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout=self.dropout_ff,
                activation="relu",
            )
        else:
            raise ValueError(f"Unrecognized Encoder `{self.encoder_arch}`")
        assert hasattr(encoder, "emb_dim")
        self.agent = Agent(
            encoder=encoder,
            d_inp=input_dim,
            d_emb=encoder.emb_dim,
            d_action=action_dim,
            policy=self.policy_dist,
            gmm_modes=self.gmm_modes,
            ignore_rtg=self.ignore_rtg,
        )
        self.agent.to(DEVICE)

        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=self.init_learning_rate,
            weight_decay=self.l2_coeff,
        )
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

    def train_step(self, batch):
        l = self.compute_loss(batch)
        self.optimizer.zero_grad()
        l["loss"].backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        return l

    def compute_loss(self, batch):
        state_seq = batch["seq"].to(DEVICE)
        actions = batch["actions"].to(DEVICE)
        pad_mask = batch["pad_mask"].to(DEVICE)
        action_dist = self.agent(state_seq, pad_mask=pad_mask)
        safe_actions = actions.clamp(-0.999, 0.999)
        logp_a = action_dist.log_prob(safe_actions).clamp(-1e3, 1e3)
        if self.policy_dist == "gmm":
            logp_a.unsqueeze_(-1)
        valid_mask = (~pad_mask).float()
        total = valid_mask.sum()
        loss = (-logp_a * valid_mask).sum() / total
        loss_dict = {
            "loss": loss,
            "max_logp_a": logp_a.max(),
            "min_logp_a": logp_a.min(),
            "seq_max": state_seq.max(),
            "seq_min": state_seq.min(),
            "pct_valid": total / torch.numel(pad_mask),
            "grad_norm": get_grad_norm(self.agent),
        }
        return loss_dict

    def init_checkpoints(self):
        self.ckpt_dir = os.path.join(self.log_dir, "ckpts")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.epoch = 0
        self.step = 0

    def load_checkpoint(self, epoch: int):
        ckpt = torch.load(
            os.path.join(self.ckpt_dir, f"{self.run_name}_epoch_{epoch}.pt")
        )
        self.agent.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt["epoch"]
        self.step = ckpt["step"]

    def save_checkpoint(self):
        state_dict = {
            "model_state": self.agent.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
        }
        ckpt_name = f"{self.run_name}_epoch_{self.epoch}.pt"
        torch.save(state_dict, os.path.join(self.ckpt_dir, ckpt_name))

    def learn(self):
        start_epoch = self.epoch
        step = self.step

        for epoch in range(start_epoch, self.epochs):
            if (
                self.eval_interval is not None
                and self.eval_interval > 0
                and epoch % self.eval_interval == 0
            ):
                eval_results = self.evaluate(timesteps=self.eval_timesteps)

            for batch in tqdm(
                self.dloader,
                desc=f"Epoch {epoch}",
                total=len(self.dloader),
                leave=False,
                colour="green",
            ):
                log_dict = self.train_step(batch)
                self.warmup_scheduler.step(self.step)
                if self.step % self.log_interval == 0:
                    self._log(log_dict, key="train")
                self.step += 1

            if epoch % self.ckpt_interval == 0:
                self.save_checkpoint()
            self.epoch += 1
        self.save_checkpoint()

    def _log_eval_results(self, eval_results):
        avg_returns = {
            k: np.array(v["returns"]).mean() for k, v in eval_results.items()
        }
        avg_success = {
            k: np.array(v["successes"]).mean() for k, v in eval_results.items()
        }
        self._log(avg_returns, key="eval-returns")
        self._log(avg_success, key="eval-success")
        self._log(self.make_figures(eval_results), key="eval-figures")

    def evaluate(self, timesteps: int):
        self.agent.eval()
        self.envs.reset_stats()

        for step in tqdm(range(timesteps)):
            seq, seq_lengths = torch_move_pad(self.envs.sequence(), device=DEVICE)
            with torch.no_grad():
                action = self.agent.get_action(
                    state_seq=seq,
                    sample_action=self.sample_during_eval,
                    seq_lengths=seq_lengths,
                )
            state, reward, done, info = self.envs.step(action)
            if self.render:
                self.envs.render()

        results = {rtg: {"returns": [], "successes": []} for rtg in self.target_returns}
        for target_rtg, returns, successes in zip(
            self.target_returns, self.envs.return_history, self.envs.success_history
        ):
            results[target_rtg]["returns"].extend(returns)
            results[target_rtg]["successes"].extend(successes)
        self._log_eval_results(results)
        self.agent.train()
        return results

    def make_figures(self, eval_dict: dict):
        # TODO?
        if not self.log_to_wandb:
            return {}
        else:
            return {}
        """
        num_plots = min(self.batch_size, 3)
        fig = plt.figure()
        ax = plt.axes()
        figs = {"RTG Curve": wandb.Image(fig)}
        plt.close()
        return figs
        """

    def _log(self, metrics_dict: dict, key: str):
        log_dict = {}
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    log_dict[k] = v.detach().cpu().item()
            else:
                log_dict[k] = v

        if self.log_to_wandb:
            wandb.log({f"{key}/{subkey}": val for subkey, val in log_dict.items()})
        else:
            print(log_dict)


if __name__ == "__main__":
    c = create_parser().parse_args()

    dt = Experiment(
        # Core
        run_name=c.run_name,
        quality=c.quality,
        task=c.task,
        log_to_wandb=c.log_to_wandb,
        # Misc.
        log_interval=c.log_interval,
        ckpt_interval=c.ckpt_interval,
        log_dir=c.log_dir,
        dloader_workers=c.dloader_workers,
        # Optimization
        warmup_steps=c.warmup_steps,
        init_learning_rate=c.init_learning_rate,
        batch_size=c.batch_size,
        epochs=c.epochs,
        l2_coeff=c.l2_coeff,
        grad_clip=c.grad_clip,
        # Eval
        eval_interval=c.eval_interval,
        sample_during_eval=not c.argmax_actions_eval,
        eval_timesteps=c.eval_timesteps,
        parallel_envs=c.parallel_envs,
        max_rollout_length=c.max_rollout_length,
        rtg_strategy=c.rtg_strategy,
        # Model
        encoder_arch=c.encoder_arch,
        context_length=c.context_length,
        ignore_rtg=c.ignore_rtg,
        d_model=c.d_model,
        d_ff=c.d_ff,
        n_heads=c.n_heads,
        layers=c.layers,
        dropout_emb=c.dropout_emb,
        dropout_ff=c.dropout_ff,
        policy_dist=c.policy_dist,
        gmm_modes=c.gmm_modes,
        render=c.render,
        async_envs=not c.render,
        hparams=vars(c),
    )
    dt.start()

    if c.ckpt is not None:
        dt.load_checkpoint(c.ckpt)

    if not c.eval:
        dt.learn()

    dt.evaluate(timesteps=10_000)
