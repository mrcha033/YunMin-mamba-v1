import argparse
from train import AdaptiveMambaTrainer, TrainingConfig, SimpleDataset, PEFT_AVAILABLE


def parse_args():
    parser = argparse.ArgumentParser(description="Training utility for YunMin Mamba")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument("--ia3", action="store_true", help="enable IA\xb3 scaling modules")
    parser.add_argument("--output-dir", default="./training_outputs", help="directory to save outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        batch_size=8,
        num_epochs=args.epochs,
        max_seq_length=64,
        learning_rate=1e-4,
        enable_masking=True,
        enable_peft=PEFT_AVAILABLE,
        enable_ia3=args.ia3,
        log_interval=10,
        eval_interval=50,
        save_interval=100,
        output_dir=args.output_dir,
    )

    train_dataset = SimpleDataset(config.vocab_size, config.max_seq_length, num_samples=1000)
    eval_dataset = SimpleDataset(config.vocab_size, config.max_seq_length, num_samples=200)

    trainer = AdaptiveMambaTrainer(config)
    trainer.train(train_dataset, eval_dataset)


if __name__ == "__main__":
    main()
