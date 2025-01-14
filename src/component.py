import csv
import pandas as pd
import logging

from huggingface_hub import login, HfApi
from keboola.component.base import ComponentBase
from keboola.component.exceptions import UserException
from configuration import Configuration
from datasets import Dataset, DatasetDict

class Component(ComponentBase):
    def __init__(self):
        super().__init__()
        self._configuration = None

    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)

    def run(self):
        self.init_configuration()

        dataset_name = self._configuration.data_name
        hf_file_path = self._configuration.file_path
        hf_username = self._configuration.user_name
        HF_TOKEN = self._configuration.pswd_hf_token

        # Change to filepath struct like expected input
        if hf_file_path:
            hf_full_path = f"{hf_username}/{hf_file_path}/{dataset_name}"
        else:
            hf_full_path = f"{hf_username}/{dataset_name}"

        login(HF_TOKEN)

        input_tables = self.get_input_tables_definitions()
        # input_files = self.get_input_files_definitions()

        if len(input_tables) == 0:
            raise UserException("No inputs found")
            
        input_table = input_tables[0]
        # data = []

        # # Read the input table
        # with open(input_table.full_path, mode='r', encoding='utf-8') as inp_file:
        #     reader = csv.DictReader(inp_file)
        #     for row in reader:
        #         data.append({key: value.strip() for key, value in row.items()})

        # Transform the data into a Hugging Face data dict
        Dataset.from_csv(input_table.full_path).push_to_hub(hf_full_path)
    #     hf_dataset = Dataset.from_pandas(pd.DataFrame(data))
    #     hf_dataset = DatasetDict({"train": hf_dataset})
       
    #    # Push to hub
    #     hf_dataset.push_to_hub(hf_full_path)
        
"""
        Main entrypoint
"""
if __name__ == "__main__":
    try:
        comp = Component()
        # this triggers the run method by default and is controlled by the configuration.action parameter
        comp.execute_action()
    except UserException as exc:
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        logging.exception(exc)
        exit(2)
