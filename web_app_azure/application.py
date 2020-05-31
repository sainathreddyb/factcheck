from flask import Flask, request, redirect
import json
import requests
import os
from azureml.core.authentication import InteractiveLoginAuthentication
#azuremlfunct

os.system('pip install pywin32-227-cp37-cp37m-win_amd64.whl')

app = Flask(__name__)

PAGE_ACCESS_TOKEN= 'EAAKbpfym8ogBAJskb7yZCxW5oL2jLroXDMYOVOHGojdtejQpIqvxbGZBCLgpyYcXQ6MFTobYGY3mZAXCl6Ln65iwDPSR2ZBKCRgZBhebRZCefrbniZCcYR6yeQCjAXTsd95tZBNpyp8omXw75kCYntstSFNwkkfZCuWwMlKvNRK0qWaz9BZCZApPINOz2yTlh3bu5EZD'

FACEBOOK_GRAPH_URL = 'https://graph.facebook.com/v2.6/me/messages'

class Bot(object):

  def __init__(self, access_token, api_url=FACEBOOK_GRAPH_URL):

      self.access_token = access_token
      self.api_url = api_url

  def send_text_message(self, psid, message, messaging_type ="RESPONSE"):

      headers = {

      'Content-Type':'application/json'

      }

      data = {

      'messaging_type':messaging_type,
      'recipient':{'id':psid},
      'message':{'text':message}

      }

      params = {'access_token':self.access_token}
      #self.api_url = self.api_url + 'messages'
      response = requests.post(self.api_url, headers = headers, params = params, data = json.dumps(data))

      print(response.content)


def query_confidence(query):

  # Get a token to authenticate to the compute instance from remote
  interactive_auth = InteractiveLoginAuthentication()
  auth_header = interactive_auth.get_authentication_header()


  headers = auth_header
  # Add content type header
  headers.update({'Content-Type':'application/json'})

  # Sample data to send to the service

  test_sample = bytes(query,encoding = 'utf8')

  # Replace with the URL for your compute instance, as determined from the previous section
  service_url = "https://fact-check-8890.eastus.instances.azureml.net/score"

  # for a compute instance, the url would be https://vm-name-6789.northcentralus.instances.azureml.net/score
  resp = requests.post(service_url, test_sample, headers=headers)
  #print("prediction:", resp.text)

  response = [float(i) for i in list(((resp.text[2:])[:-2]).split(','))]

  return response


@app.route("/")
def hello():
  return "Hello Reubs!"


@app.route('/fb', methods=['GET','POST'])
def webhook():
  print("in webhook")
  if request.method == 'GET':
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')

    if token=='secret':
        return str(challenge)

    else:
        return 400
        #this int return will cause errors when running on localhost:5000

  else:

    #{"object":"page","entry":[{"id":"103772971365001","time":1590831512511,"messaging":[{"sender":{"id":"3013672428751081"},"recipient":{"id":"103772971365001"},"timestamp":1590831512154,"message":{"mid":"m_oQASQ8lZMeAyFPj9fYAaDOdUitW4nGbO6Ay5AyfJKo8b3iO6-zPu1_LW8jUwjQJbclvP5u9Lvs-GqWQPKEH5aw","text":"hellothere"}}]}]}
    
    print(request.data)

    data = json.loads(request.data)
    messaging_events = data['entry'][0]['messaging']
    bot = Bot(PAGE_ACCESS_TOKEN)

    for message in messaging_events:

        user_id = message['sender']['id']
        text_input = message['message'].get('text')

        print('Message from user ID {} is : {}'.format(user_id,text_input))

        response = query_confidence(text_input)
        if response[1]>=0.6:
            val='is likely to be correct'
        elif response[1]>=0.3 and response[1]<0.6:
            val='may or may not be correct'
        else:
            val='is likely to be incorrect'

        bot.send_text_message(user_id,'On a scale of [0-1] with 1 being definietly true\n\nYour query has a confidence score of :'+str("%.3f"%response[1])+'\nthis means your query '+val)

    return '200'



if __name__ == "__main__":
  app.run(debug=True)