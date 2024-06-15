


#!pip install python-dotenv
 

from dotenv import load_dotenv
import os
load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

 
print(os.getenv("ACCESS_TOKEN"))

