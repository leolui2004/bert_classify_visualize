import sqlite3
import os
import re

DIR = 'livedoor/'
DS = f'{DIR}ldcc-20140209.tar/text/'
DB = f'{DIR}livedoor.db'
category = ['it-life-hack', 'kaden-channel', 'movie-enter', 'peachy', 'smax', 'sports-watch']

class database:
    def __init__(self, DS, DB):
        self.DS = DS
        self.DB = DB
    
    def create(self):
        conn = sqlite3.connect(self.DB)
        c = conn.cursor()
        c.execute('''CREATE TABLE topic(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            label TEXT NOT NULL);''')
        conn.commit()
        conn.close()
    
    def save(self):
        for cat in range(len(category)):
            dir = f'{DS}{category[cat]}/'
            file_list = os.listdir(path=dir)
            for file in file_list:
                f = open(f'{dir}{file}', 'r', encoding='utf-8_sig')
                line = 0
                for i in f:
                    line += 1
                    if line == 3:
                        text = re.sub(r'【.*?】', '', str(i).replace('\n',''))
                        
                        if len(text) > 0:
                            conn = sqlite3.connect(self.DB)
                            c = conn.cursor()
                            c.execute("INSERT INTO topic VALUES (NULL,?,?)", (text,cat));
                            conn.commit()
                            conn.close()
            
            print(f'Category {category[cat]} Finished')

dbs = database(DS,DB)
dbs.create()
dbs.save()