#!/usr/bin/env python
# coding: utf-8

# In[1]:


from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("pytorch")
myenv.add_conda_package("torchvision")
myenv.add_channel("pytorch")
myenv.add_pip_package("transformers")
myenv.add_pip_package("tokenizers")

env_file = "env_pytorch.yml"

with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)

with open(env_file,"r") as f:
    print(f.read())


# In[ ]:





# In[2]:


from azureml.core.environment import Environment


# In[3]:


myenv = Environment.from_conda_specification(name = "myenv",
                                             file_path = "env_pytorch.yml")


# In[4]:


from azureml.core import Workspace
ws = Workspace.from_config(path="config")


# In[14]:


from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
inference_config = InferenceConfig(entry_script='score.py',
                                   environment=myenv)


# In[15]:


from azureml.core.model import InferenceConfig, Model


# In[16]:


new_model = Model.register(model_path="bert_files",
                           model_name="trail",
                           tags={"key": "0.1"},
                           description="test",
                           workspace=ws)


# In[17]:


from azureml.core.webservice import LocalWebservice, Webservice

deployment_config = LocalWebservice.deploy_configuration(port=8890)
service = Model.deploy(ws, "myservice", [new_model], inference_config, deployment_config)
service.wait_for_deployment(show_output = True)
print(service.state)


# In[12]:


import requests
import json
from azureml.core.authentication import InteractiveLoginAuthentication

# Get a token to authenticate to the compute instance from remote
interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()


headers = auth_header
# Add content type header
headers.update({'Content-Type':'application/json'})

# Sample data to send to the service

test_sample = bytes("",encoding = 'utf8')

# Replace with the URL for your compute instance, as determined from the previous section
service_url = "https://factcheck-8890.eastus.instances.azureml.net/score"
# for a compute instance, the url would be https://vm-name-6789.northcentralus.instances.azureml.net/score
resp = requests.post(service_url, test_sample, headers=headers)
print("prediction:", resp.text)


# In[10]:


get_ipython().system('pip install kaggle')


# In[11]:


os.environ['KAGGLE_USERNAME'] = 'sainathreddy'
os.environ['KAGGLE_KEY'] = '5154625efff2bfc88696dd0f25615e25'


# In[13]:


get_ipython().system('kaggle kernels output sainathreddy/bert-base-uncased-using-pytorch -p bert_files')


# In[ ]:




