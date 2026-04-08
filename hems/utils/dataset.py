# hems/utils/dataset.py
from pathlib import Path
from typing import Dict

class DataSet:
    """Manages dataset paths and schema files for CityLearn challenge datasets.
    
    This class provides utilities to locate and validate schema files for different
    CityLearn challenge datasets. It maintains a centralized mapping of dataset names
    to their corresponding schema file paths.
    
    Attributes:
        PROJECT_ROOT (Path): The root directory of the project, resolved from the
            current file's location (three levels up).
        DATASET_DIR (Path): The directory containing all citylearn datasets.
        SCHEMA_MAP (dict): Mapping of dataset names to their relative schema file paths.
    """

    # Resolve project root by going three levels up from this file
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

    # Directory containing all CityLearn datasets
    DATASET_DIR: Path = PROJECT_ROOT / "datasets" / "citylearn_datasets"

    # Default dataset name
    DEFAULT_DATASET: str = "phase_all"

    # Mapping of dataset names to their schema file paths
    SCHEMA_MAP: Dict[str, str] = {
        "phase_all": "citylearn_challenge_2022_phase_all/schema.json",
        "phase_1":   "citylearn_challenge_2022_phase_1/schema.json",
        "demo_1":    "demo_1/schema.json",
    }

    def get_schema(self, dataset_name: str = None) -> Path:
        """Retrieves the full path to a dataset's schema file.
        
        Validates that the dataset name exists in the schema map and that the
        corresponding schema file exists on disk.
        
        Args:
            dataset_name (str): The name of the dataset to get the schema for.
                Must be one of the keys in SCHEMA_MAP.
        
        Returns:
            Path: The full path to the schema file.
        
        Raises:
            ValueError: If the dataset_name is not found in SCHEMA_MAP.
            FileNotFoundError: If the schema file does not exist at the expected path.
        
        Example:
            >>> dataset = DataSet()
            >>> schema_path = dataset.get_schema("phase_1")
            >>> print(schema_path)
            /path/to/project/datasets/citylearn_datasets/citylearn_challenge_2022_phase_1/schema.json
        """
        
        # Set a default dataset name if None
        if dataset_name is None:
            dataset_name = self.DEFAULT_DATASET

        # Validate that the dataset name is known
        if dataset_name not in self.SCHEMA_MAP:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Construct the full path to the schema file
        path = self.DATASET_DIR / self.SCHEMA_MAP[dataset_name]
        
        # Validate that the schema file exists
        if not path.is_file():
            raise FileNotFoundError(f"Schema not found: {path}")
        
        return path
