### Getting Started
Setup the conda env using:
```
source setup.sh
```
If conda is not installed on the system, you can use the `./install_conda.sh` script in the `./conda/` dir. 

Before running the code, make sure to put to copy the dataset to the `./datasets/` dir.
> [!TIP]
> You can directly copy using `cp` or symbolic links `ln -s /path/to/R0_DATA_FLEX_F1 ./datasets/`. \

Then run the code using:
```
python main.py
```
