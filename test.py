from wikidata.client import Client

client = Client()
entity = client.get('Q5', load=True)
print(entity)