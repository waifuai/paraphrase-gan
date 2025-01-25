import pathlib
import shutil
from typing import Dict, Any
import logging as log

class DataPreparer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_dir = pathlib.Path(config['input_dir'])
        self.output_dir = pathlib.Path(config['output_dir'])
        
    def prepare(self):
        self._validate_inputs()
        self._clean_output()
        self._process_files()
        
    def _validate_inputs(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} not found")
            
    def _clean_output(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        
    def _process_files(self):
        # Unified processing logic
        for file in self.input_dir.glob('*.tsv'):
            self._process_file(file)
            
    def _process_file(self, file_path):
        output_file = self.output_dir / file_path.name
        
        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                # Basic cleaning and normalization
                cleaned = line.strip().lower()
                if not cleaned or len(cleaned.split('\t')) < 2:
                    continue
                
                # Add additional processing as needed
                outfile.write(f"{cleaned}\n")
        
        log.info(f"Processed {file_path.name} -> {output_file.name}")

if __name__ == "__main__":
    config = {
        'input_dir': 'data/raw',
        'output_dir': 'data/processed/default'
    }
    preparer = DataPreparer(config)
    preparer.prepare()
    print("Data preparation complete.")