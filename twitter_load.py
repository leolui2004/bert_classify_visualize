import pandas as pd
import sqlite3

class SaveLoad:
    def __init__(self, DataPath, DBPath):
        self.DataPath = DataPath
        self.DBPath = DBPath
    
    def Database(self):
        conn = sqlite3.connect(self.DBPath)
        c = conn.cursor()
        c.execute('''CREATE TABLE tw(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            text TEXT NOT NULL,
            label TEXT NOT NULL);''')
        conn.commit()
        conn.close()
    
    def LoadFile(self):
        object = pd.read_pickle(self.DataPath)
        for i in range(len(object)):
            if object[i]['text'] != '':
                
                _list = object[i]['label']
                label = ''.join(map(str, _list))
                
                conn = sqlite3.connect(self.DBPath)
                c = conn.cursor()
                c.execute("INSERT INTO tw (id,topic,text,label) \
                          VALUES (NULL,?,?,?)", (object[i]['topic'], object[i]['text'], label));
                conn.commit()
                conn.close()
                
                if i % 10000 == 0:
                    print(i, len(object))

DIR = 'twitter/'
DATA = f'{DIR}twitterJSA_data.pickle'
DB = f'{DIR}twitter.db'

saveload = SaveLoad(DATA, DB)
saveload.Database()
saveload.LoadFile()