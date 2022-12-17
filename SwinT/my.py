import pandas as pd
x = "/home/lzz/yueyubiao/paper_program/granulocyte_cell/val_data/BAN/123.jpg"
x = x.split("/")
print(x[-2])
# xx = []
# for x in range(1,769):
#     x = str(x)
#     xx.append(x)
# xx = tuple(xx)
# columns = ("wenjianlujing", "zhenshifenlei","yucefenlei","fenleigailv","BAN","BAS","EOS","SEG")
# columns = columns + xx
# print(columns)
# res = pd.DataFrame(columns=("文本路徑","這是分類"))
#
# res.loc[0] = ["11","22"]
# res.loc[1] = ["22","33"]
#
# res.to_excel("test.xlsx")
