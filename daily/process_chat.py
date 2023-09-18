file = open('chatwang.txt', encoding='utf-8', errors='ignore')
lines = file.readlines()
index = 0
index2 = 0
temp = []
for line in lines:
    if index == 5000:
        index = 0
        save = open(f'chatwang{index2}.txt', 'a+', encoding='utf-8', errors='ignore')
        save.writelines(temp)
        index2 += 1
        temp=[]
    if line == '\n':
        continue
    temp.append(line)
    index += 1
