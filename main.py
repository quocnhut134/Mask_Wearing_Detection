from src.data_processor import DataProcessor
from src.trainer import MaskTrainer
import os
from src import config

def main():
    processor = DataProcessor()
    processor.process_data(split_ratio=0.8)

    trainer = MaskTrainer()
    trainer.train()

    trainer.validate()

if __name__ == "__main__":
    main()