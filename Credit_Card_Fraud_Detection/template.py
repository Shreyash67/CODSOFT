import os
from pathlib import Path

package_name="CreditCardFraudDetection"

list_of_files=[
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/components/data_ingestion.py",
    f"src/{package_name}/components/data_transformation.py",
    f"src/{package_name}/components/model_train.py",
    f"src/{package_name}/pipelines/__init__.py",
    f"src/{package_name}/pipelines/model_evalution.py",
    f"src/{package_name}/logger.py",
    "notebooks/EDA.ipynb",
    "notebooks/FE.ipynb",
    "requirements.txt",
    "setup.py",
]


# here will create a directory

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    
    """ how exist_ok works:if "directory" already exists, 
    os.makedirs() will not raise an error, and it will do nothing. 
    If "my_directory" doesn't exist, it will create the directory.
    """
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,"w") as f:
            pass
    else:
        print("file already exists")

# here will use the file handling