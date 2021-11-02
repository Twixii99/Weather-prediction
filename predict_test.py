import requests

url = 'http://localhost:9696/predict'
weather = {
'location': 'albury',
'mintemp': 5.9,
'maxtemp': 16.7,
'rainfall': 0.0,
'evaporation': 5.0,
'sunshine': 8.546358,
'windgustdir': 'wnw',
'windgustspeed': 39.0,
'winddir9am': 'wnw',
'winddir3pm': 'wsw',
'windspeed9am': 13.0,
'windspeed3pm': 22.0,
'humidity9am': 83.0,
'humidity3pm': 47.0,
'pressure9am': 1013.9,
'pressure3pm': 1012.7,
'cloud9am': 7.0,
'cloud3pm': 4.0,
'temp9am': 9.9,
'temp3pm': 15.0,
'raintoday': 0
}

response = requests.post(url, json=weather).json()
print(response)

if response['Will it rain ?'] == True:
    print('It will rain tomorrow')
else:
    print('It won\'t rain tomorrow')