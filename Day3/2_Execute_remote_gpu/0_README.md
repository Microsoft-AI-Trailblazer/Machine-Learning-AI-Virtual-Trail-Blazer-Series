# Training a Convolutional Neural Network on a GPU-enabled Virtual Machine

This hands-on lab guides you with how to configure Azure Machine Learning to use a Data Science Virtual Machine (DSVM) equipped with GPUs as an execution target. By now, you should be familiar with how to execute locally (with and without docker).

***NOTE:*** There are several pre-requisites for this course, including an understanding and implementation of: 
  *  Machine Learning and Data Science
  *  Convolutional Neural Networks
  *  Intermediate to Advanced Python programming
  *  Familiarity with Docker containers

## Background

Sentiment analysis is a well-known task in the realm of natural language processing (NLP), and it aims to determine the attitude of a speaker/writer. Frequently, artificial neural networks (and deep learning) are used to estimate such sentiment. In this lab, we will use this approach. In the [resources](resources) folder, there is a `sentiment_reviews.py` script that builds a neural network to predict sentiment from IMDB movie reviews. The script uses [keras](https://keras.io/) with [tensorflow](https://www.tensorflow.org/) as the backend. It is part
of a larger [real-world example](https://docs.microsoft.com/en-us/azure/machine-learning/preview/scenario-sentiment-analysis-deep-learning).

## 1. Setup

A. Create a new blank project, and call it 'sentiment-gpu'.

B. From the resources folder, copy `sentiment_analysis.py` to the project folder.

C. Review the `sentiment_analysis.py` file. Take particular note of the following:
  - The `keras` dependencies loaded at the top of the file
  - The `build_model()` method that both constructs the architecture of the network and fits it.
  - The last 20 lines, which correspond to the control flow of what the script does when executed.

Once you are comfortable with this script, we will execute this on a remote VM with GPU hardware.

## 2. Remote Execution on a DSVM equipped with GPU

Our first step is to make sure we have access to a VM with a GPU.

### 2.1 Create a Ubuntu-based Linux Data Science Virtual Machine in Azure

A. Open your web browser and go to the [Azure portal](https://portal.azure.com/)

B. Select `+ New` on the left of the portal.
Search for `Data Science Virtual Machine for Linux Ubuntu CSP` in the marketplace. Choosing **Ubuntu** is critical.

C. Click Create to create an Ubuntu DSVM.

D. Fill in the `Basics` blade with the required information. When selecting the location for your VM, note that GPU VMs (e.g. `NC-series`) are only available in certain Azure regions, for example, South Central US. See [compute products available by region](https://azure.microsoft.com/en-us/regions/services/). Click OK to save the Basics information.

E. Choose the size of the virtual machine. Select one of the sizes with NC-prefixed VMs, which are equipped with NVidia GPU chips. Click **View All** to see the full list as needed. Learn more about [GPU-equipped Azure VMs](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu).

F. Finish the remaining settings and review the purchase information. Click **Purchase** to create the VM. Take note of the IP address allocated to the virtual machine - you will need this (or a domain name) in the next section when you are configuring AML. 

### 2.2 Create a new Compute Target

A. With the new project called 'sentiment-gpu' open in Workbench, launch the command line. 

B. Enter the following command. Replace the placeholder text from the example below with your own values for the name, IP address, username, and password. 

```az ml computetarget attach remotedocker --name <COMPUTETARGETNAME> --address <DSVM_IPADDRESS> --username <USERNAME> --password <PASSWD>```

For example, your command could look like:

```az ml computetarget attach remotedocker --name myNCdsvm --address 127.0.0.1 --username dsvmuser --password myterriblepassword123```

This will create two files in the `aml_config` directory associated with the project. Specifically, within that directory, it will create `<COMPUTETARGETNAME>.runconfig` and `<COMPUTETARGETNAME>.compute` files for the computetarget you just created. Take a moment to look at those two files.

### 2.3 Configure the compute target to leverage GPU compute.

In order to run the script on a remote VM with GPU support, we need to edit three files: 

- `aml_config/conda_dependencies.yml` to include the python dependencies.
- `<COMPUTETARGETNAME>.compute>` to make sure that the docker image that will be created can support GPU execution.
- `<COMPUTETARGETNAME>.runconfig>` to make sure that the runtime environment is python.

From Workbench, open File View, and hit the Refresh button. Navigate to the `aml_config` directory, and find the `conda_dependencies.yml`, `<COMPUTETARGETNAME>.compute`, and `<COMPUTETARGETNAME>.runconfig` files.


A. Edit `conda_dependencies.yml`. This file is referenced in `<COMPUTETARGETNAME>.runconfig` and specifies the python dependencies that we need to have installed on the compute target. We need to include the deep learning packages (`tensorflow-gpu` and `keras`) as dependencies that must be managed. The best way to include the `tensorflow-gpu` package is to include the specific version available in the `anaconda` channel. Once we add these dependencies, the `conda_dependencies.yml` should look as follows:

```
name: sentiment-gpu-project
channels:
  - defaults
  - anaconda
dependencies:
  - python=3.5.2
  - ipykernel=4.6.1
  - tensorflow-gpu=1.4.1
  - pip:
    - keras==2.1.4
    # Required packages for AzureML execution, history, and data preparation.
    - --index-url https://azuremldownloads.azureedge.net/python-repository/preview
    - --extra-index-url https://pypi.python.org/simple
    - azureml-requirements
    # The API for Azure Machine Learning Model Management Service.
    # Details: https://github.com/Azure/Machine-Learning-Operationalization
    - azure-ml-api-sdk==0.1.0a10  
```


B. Open the `<COMPUTETARGETNAME>.compute` file and make two changes:

- Change the `baseDockerImage` value to `microsoft/mmlspark:plus-gpu-0.9.9` 
- Add a new line `nvidiaDocker: true`. The file should have these two lines:

```
baseDockerImage: microsoft/mmlspark:plus-gpu-0.9.9
nvidiaDocker: true
```

C. Open the `<COMPUTETARGETNAME>.runconfig` file, and make one change:

- Change the `Framework` value from `Pyspark` to `Python`. 

Once these values are changed, we can then prepare the compute environment on the remote VM.

D. Run the prepare command 

```az ml experiment prepare -c <COMPUTETARGETNAME>```

In the prior example, it would look like:

```az ml experiment prepare -c myNCdsvm```

This will take a little time (5-10 minutes).

### 2.4 Execute on the remote VM

Once this has successfully completed, you can run `sentiment_reviews.py` script from the command line:

`az ml experiment submit -c <COMPUTETARGETNAME> sentiment_reviews.py`

To verify that the GPU is used, examine the run output to see something like the following:

```
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID 5884:00:00.0
Total memory: 11.17GiB Free memory: 11.10GiB
2018-02-01 19:55:58.353261: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 5884:00:00.0)
```

If this succeeded, then congratulations, you have just executed on a remote VM using GPU compute. 

This concludes this lab. If you would like to examine how to operationalize this model in a scoring service, you can view the [real-world example](https://docs.microsoft.com/en-us/azure/machine-learning/preview/scenario-sentiment-analysis-deep-learning).