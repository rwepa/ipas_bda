"""
#          : iPAS-basic.py
# Title    : iPAS-初級參考資料.py
# Author   : Ming-Chang Lee
# Date     : 2023.04.06
# YouTube  : https://www.youtube.com/@alan9956
# RWEPA    : http://rwepa.blogspot.tw/
# GitHub   : https://github.com/rwepa
# Email    : alan9956@gmail.com
"""

"""
初級巨量資料分析師-109-01.20-樣題L11-科目1-資料導向程式設計.pdf
考試須準備筆, 方便計算.
"""

##################################################
# 1. 矩陣中，線性獨立的行向量或線性獨立的列向量稱為？
# (A) 相關矩陣（Correlation matrix）
# (B) 單位矩陣（Identity matrix）
# (C) 秩（Rank）
# (D) 反矩陣（Inverse matrix）

# 答案: C

# 相關矩陣表示相關係數, 
# 單位矩陣是n階單位矩陣，是一個n×n的方形矩陣，其主對角線元素為1，其餘元素為0。
# https://en.wikipedia.org/wiki/Identity_matrix

# 在線性代數中，一個矩陣 A 的行秩是 A 的線性獨立的縱行的極大數目。類似地，列秩是 A 的線性獨立的橫列的極大數目。
# 矩陣的行秩和列秩總是相等的，因此它們可以簡單地稱作矩陣 A 的秩。通常表示為 rank(A)。
# https://en.wikipedia.org/wiki/Rank_(linear_algebra)

# 逆矩陣（inverse matrix），又稱乘法反方陣、反矩陣。在線性代數中，給定一個n 階方陣 A，若存在一n 階方陣 B，
# 使得 AB=BA=I_n，其中 I_n為n 階單位矩陣，則稱 A 是可逆的，且 B 是 A 的逆矩陣，記作 A^[-1]。
# 若方陣 A 的逆矩陣存在，則稱 A 為非奇異方陣（nonsingular matrix）或非退化方陣（nondegenerate matrix）或可逆方陣（inverse matrix）。
# https://en.wikipedia.org/wiki/Invertible_matrix

##################################################
# 2. 在 Python 語言中，已知「a=numpy.array([[1, 3],[2, 4]])」，則 3*a 意義為何？

# 答案： B

# (B)	3 乘以每個元素
# 同理，b=[1,2,3], 練習 3*b 之結果。

##################################################
# 7. 考慮 Python 語言運算「a=list(range(3));b=list(range(4));c=[a,b]」，則 「c[1][2]」結果為何？
# 理解其意義。

##################################################
# 13.	在資料表正規化的過程（1NF 到 BCNF）中，每一個階段都是以欄位的「相依性」作為分割資料表的依據之一，關於正規化步驟之敘述，下列何者不正確？
# (A) 第一正規化型式：除去重覆群
# (B) 第二正規化型式：除去部分相依
# (C) 第三正規化型式：除去遞移相依
# (D) Boyce-Code 正規化型式：除去多值相依

# 答案： D

# Boyce-Codd 正規化
# 1. 符合 2NF 的格式
# 2. 各欄位與 Primary Key 沒有『間接相依』的關係
# 3. Primary Key 中的各欄位不可以相依於其他非Primary Key 的欄位

##################################################
# 27. 以下為 Java 的物件建構流程，請問其排列順序應為何？
# 1.	初始化
# 2.	儲存物件參考
# 3.	實體化
# 4.	宣告
# 5.	執行建構式
# (A)  41235
# (B)  41253
# (C)  43152
# (D)  43125

# 答案： C

# Java 的物件建構流程，請問其排列順序:
# 宣告 \ 實體化 \ 初始化 \ 執行建構式 \ 儲存物件參考

"""
初級巨量資料分析師-109-01.20-樣題L12-科目2-資料處理與分析概論.pdf
"""

##################################################
# 10. 下列哪個方法是將時間序列資料轉換到頻域空間？
# (A) 傅立葉轉換
# (B) 特徵值加權
# (C) 資料降維
# (D) 隨機抽樣

# 答案： A

# 傅立葉轉換（法語：Transformation de Fourier，英語：Fourier transform，縮寫：FT）是一種線性積分轉換，
# 用於函數（應用上稱作「信號」）在時域和頻域之間的轉換。因其基本思想首先由法國學者約瑟夫·傅立葉系統地提出，所以以其名字來命名以示紀念。
# 傅立葉轉換在物理學和工程學中有許多應用。傅立葉轉換的作用是將函數分解為不同特徵的正弦函數的和，
# 如同化學分析來分析一個化合物的元素成分。對於一個函數，也可對其進行分析，來確定組成它的基本（正弦函數）成分。
# https://en.wikipedia.org/wiki/Fourier_transform

##################################################
# 14. 下列哪種方法不屬於特徵選擇（Feature-Selection）的標準方法？
# (A) 嵌入方法（Embedded）
# (B) 過濾方法（Filter）
# (C) 包裝方法（Wrapper）
# (D) 抽樣方法（Sampling）

# 答案： D

# 特徵選擇：
# 包裝法 (Wrapper) - 所有特徵子集皆測試看看
# 過濾法 (Filter) - 使用指標評估哪些特徵子集較佳，計算較包裝法快速。
# 嵌入法 (Embedded method) - 模型中嵌入L1, L2技術
# 計算複雜度：
# 過濾法≤"嵌入法"≤"包裝法" (計算較複雜，時間較久)

##################################################
# 17. 關於 MapReduce 框架，下列敘述何者不正確？
# (A) Mapper 的輸出需要是鍵值組（key-value pair）的結構
# (B) 實現 Reducer，通常是定義如何處理個別鍵值下的值集合
# (C) Reducer 的輸出值通常也是鍵值組（key-value pair）的結構
# (D) 資料在進入 Map 階段之前會經過整理階段（shuffle）

# 答案： D

# Input \ Splitting \ Mappping \ Shuffling  \ Reducing \ Final result
# 資料在進入 Map 階段之前會經過分割階段（Splitting）

##################################################
# 22. 若兩事件 X、Y 為某試驗可能發生之二獨立事件，P(X)>0，P(Y)>0，下列何者不正確？
# (A) P(X∪Y)=P(X)+P(Y)
# (B) P(X|Y)=P(X)
# (C) P(X|Y)P(Y)=P(Y|X)P(X)
# (D) P(X∩Y)=P(X)P(Y)

# 答案： A

# P(X∪Y)=P(X)+P(Y)-P(X∩Y)

##################################################
# 29. 關於模型績效評估，下列敘述何者不正確？
# (A) 殘差（或稱預測誤差）是預測的反應變數值減去真實的反應變數值
# (B) 迴歸模型績效衡量大多基於殘差
# (C) 赤池弘次訊息準則（Akaike’s Information Criterion, AIC）與舒瓦茲貝氏訊息準則
# （Schwarz’s Bayesian Information Criterion, BIC）的不同在於懲罰過多變數入模的方式不同
# (D) Mallow’s Cp 準則有考慮建模所用的變數數量，因此適合用來比較不同變數數量下的模型績效

# 答案： A

# 殘差（或稱預測誤差）= 真實的反應變數值 - 預測的反應變數值

##################################################
# 34. 某公司員工 8 人，月薪如下：
# 編號 		#1	#2	#3	#4	#5	#6	#7	#8
# 月薪(千元)	22	25	25	28	30	30	60	100
# 下列敘述何者不正確？
# (A) 薪資中位數為 29 千元
# (B) 有 50%的員工，薪資≧第二四分位數
# (C) 有 50%的員工，薪資≧平均值
# (D) 繪製成箱形圖（Box plot，盒鬚圖），呈現右偏

# 答案： C

# 使用R:
# x <- c(22,25,25,28,30,30,60,100)
# x
# (A) median(x) = 29
# (B) 中位數
# (C) mean(x) = 40, P(薪資≧平均值)=2/8=0.25
# (D) boxplot(x, horizontal = TRUE)

"""
初級巨量資料分析師-111.11.19-11102-B11-科目1-資料導向程式設計-疑義題釋義版.pdf
"""

##################################################
# 1. 如附圖所示，使用 Python 語言定義串列（list），下列敘述哪一項正確？
a = [1, 2, 3]
b = [4, 5, 6, "HelloWorld"]
c = [a, b] # [[1, 2, 3], [4, 5, 6, 'HelloWorld']]
d = a + b #  [1, 2, 3, 4, 5, 6, 'HelloWorld']

# (A) a[-1]的結果為[2, 3]
# (B) b[1:3]的結果為[5, 6]
# (C) c[1][2]的結果為 5
# (D) d[1]的結果為 7

# 答案： B

# (A) a[-1]的結果為[2, 3]
a[-1] # 3
# (B) b[1:3]的結果為[5, 6]
b[1:3] # 答案正確
# (C) c[1][2]的結果為 5
c[1][2] # 6
# (D) d[1]的結果為 7
d[1] # 2

##################################################
# 3. 如附圖所示，己知某企業自 2020 年第 2 季起最近 5 期的產品銷售量（單位為百
# 萬元）為{50, 40, 60, 90, 70}，以 R 語言建立 myts 時間序列物件（Time-series
# Objects），下列敘述哪一項正確？

# 答案： C

# 使用R:
# myts <- ts(c(50,40,60,90,70), start = c(2020,2), frequency = 4)
# > myts
#      Qtr1 Qtr2 Qtr3 Qtr4
# 2020        50   40   60
# 2021   90   70          
# class(myts)
# [1] "ts"
# str(myts)
#  Time-Series [1:5] from 2020 to 2021: 50 40 60 90 70

##################################################
# 4. 如附圖所示為 R 語言程式碼片段，下列敘述哪一項正確？

# 答案： C

# 本題為熟悉R之矩陣(matrix)操作,練習本題之R程式.
# R預設矩陣之建立為直行, byrow = TRUE 表示物件是以橫列順序建立.

##################################################
# 5. 如附圖所示為 Python 程式碼片段，下列敘述哪一項正確？

city = ["台北市", "台中市", "新北市", "高雄市", "台北市"]
mycity1, mycity2, *mycity3 = city

# (A)	type(mycity1)結果是 list
# (B)	mycity1 有 3 個元素
# (C)	mycity1 == mycity2 結果為 True
# (D)	mycity3[2] == mycity1 結果為 True

type(mycity1) # str
mycity1 # '台北市', 有 1 個元素
mycity1 == mycity2 # 結果為 False
mycity3[2] == mycity1 # 結果為 True

##################################################
# 6. 如附圖所示為 R 語言程式碼片段，下列敘述哪一項正確？

# 答案： C

# 本題為熟悉R之資料框(data.frame)操作,練習本題之R程式.

##################################################
# 7. 如附圖所示為 Python 語言程式碼片段，其執行結果下列哪一項錯誤？

d = {
     'key1': {1,2,3,4,5},
     'key2': {1,3,5,7,9}
}

# (A)	執行 d['key3'] = 'Hello'，可於 d 中新增鍵（key）為'key3'、值（value）為'Hello'
# (B)	d['key1']的型態（type）為 set
# (C)  d['key1'][0]的值 1
# (D)  執行 d['key1'].union(d['key2'])之結果為{1, 2, 3, 4, 5, 7, 9}

# 答案： C

# 本題為熟悉 Python 之字典(dict)操作, 練習本題之 Python 程式.

d['key3'] = 'Hello' # (A) 正確
type(d['key1'])     # (B) 正確
d['key1'][0]        # (C) 錯誤 TypeError: 'set' object is not subscriptable
d['key1'].union(d['key2']) # (D) 正確

##################################################
# 8. 如附圖所示為 R 語言程式碼片段，其執行結果下列哪一項錯誤？

"""
x <- 1:5
names(x) <- c("A ", "B ", "C ", "D ", "E ")
print(x[-4])
# 結果:
#  A  B  C  E  
#  1  2  3  5 
# x[-4]表示刪除第4個元素, R指標從1開始.
"""

##################################################
# 12. 關於 HBase，下列敘述哪一項錯誤？

# (A)	HBase 進入 shell 的指令是「hbase shell」
# (B)	HBase 建立 test 表格與 info 列簇的指令是「create 'test', 'info'」
# (C)	HBase 表格中插入值的指令為 create
# (D)	HBase 顯示所有表格指令為 list

# 答案： C

# (C)	HBase 表格中插入值的正確指令為 put

##################################################
# 19. 如附圖所示， R  語言使用  read.table  函數匯入  CSV  文字檔， 
# 執行 df <- read.table("ipas.csv", sep = ",") ，下列敘述哪一項錯誤？

# (A)	class(df)結果是 "data.frame"
# (B)	nrow(df)結果是 7
# (C)	ncol(df)結果是 5
# (D)	names(df) 結 果 是	"Sepal.Length"  "Sepal.Width"  "Petal.Length"  "Petal.Width" "Species"

# 答案： D

# urls <- "https://raw.githubusercontent.com/rwepa/ipas_bda/main/data/ipas.csv"
# df <- read.table(urls, sep = ",")
# df

# 第1列名稱因為沒有註明 header = TRUE, 因此第1列名稱已經轉為資料
# names(df)
# [1] "V1" "V2" "V3" "V4" "V5"
# > df <- read.table("ipas.csv", sep = ",")
# > df
#             V1          V2           V3          V4      V5
# 1 Sepal.Length Sepal.Width Petal.Length Petal.Width Species
# 2          5.1         3.5          1.4         0.2  setosa
# 3          4.9           3          1.4         0.2  setosa
# 4          4.7         3.2          1.3         0.2  setosa
# 5          4.6         3.1          1.5         0.2  setosa
# 6            5         3.6          1.4         0.2  setosa
# 7          5.4         3.9          1.7         0.4  setosa

##################################################
# 25. 如附圖所示，R 語言使用 read.fwf 函數匯入固定寬度文字檔，圖中的紅色框線空白處須加上什麼參數？

# 答案： B

# read.fwf(file = "ipas-width.txt", width = c(4, -1, 8))
# fwf: Read Fixed Width
# c(4, -1, 8) 依序讀取4個字元, -1表示省略1個字元, 8表示讀取8個字元


# 31. 如附圖所示為 R 語言的二維矩陣，執行到下列哪一行程式碼之後，會改變維度？
# a <- matrix(1:9, nrow = 3)
# b <- matrix(10:18, nrow = 3)

# 答案： B
# 應修正為 colMeans(a)

# a+b
# colMeans(a)
# sqrt(a)
# a %*% b

##################################################
# 37. 下列哪一項Python 程式碼無法將全域變數 one 的內容從串列[1]改變成串列[2]？

# (A)
one=[1]
def a():
    one[0]=2
a()

# (B)
one=[1]
def b():
    one=[2]
b()

# (C)
one=[1]
def c():
    global one
    one[0]=2
c()

# (D)
one=[1]
def d():
    global one
    one=[2]
d()

# 答案： B

# Python 語言中, 函數使用 global 表示使用全域變數.

##################################################
# 38. 如附圖所示為 Python 程式碼片段，其執行結果下列哪一項正確？

def myappend(element, array=[]):
    array.append(element)
    return array

ans = myappend(3,[1,2])
ans = myappend(4)
ans = myappend(7,[5,6])
ans = myappend(8)
print(ans)

# (A)  [8]
# (B)  [[5,6],7,8]
# (C)  [4,8]
# (D)  [[1,2],3,4,[5,6],7,8]

# 答案： C
# 本題使用 list 物件 - append 方法.

##################################################
# 41. 如附圖所示之 Python 程式碼，其執行結果 value1、value2 的值下列哪一項正確？

def personal_info(name, *value1, **value2):
    return value1[0], value2

value1, value2 = personal_info('Oscar', '2022-01-01', 'Satyrday', gender='male', cith='Taipei')

# (A)	value1 的值為'Oscar'，value2 的值為('gender': 'male', 'city': 'Taipei')
# (B)	value1 的值為'2022-01-01'，value2 的值為('male', 'Taipei')
# (C)	value1 的值為'2022-01-01'，value2 的值為{'gender': 'male', 'city': 'Taipei'}
# (D)	value1 的值為'Saturday'，value2 的值為{'gender': 'male', 'city': 'Taipei'}

# 答案： C

# 本題為 Python 函數運用
# 'Oscar' 以關鍵字參數方式傳入 name.
# '2022-01-01', 'Satyrday' 以 tuple 方式傳入 value1.
# 'Satyrday', gender='male', cith='Taipei' 以 dict 方式傳入 value2.

# Python 函數的參數：
# 1. 關鍵字參數(Keyword Argument)：呼叫函數參數時, 在傳入參數值的前面加上函數所定義的參數名稱.
# 2. 預設值參數(Default Argument)：在函數定義的參數中, 沒有傳入參數值時，依照預設值來進行運算, 
# 其中必要參數(Required Argument)須放在預設參數(Optional Argument)的前面.
# 3. *args 運算子: 使用 tuple 傳入參數.
# 4. **kwargs 運算子: 使用 dict 傳入參數.

# Parameters vs. Arguments (from a function's perspective):
# A 'parameter' is the variable listed inside the parentheses in the function definition.
# An 'argument' is the value that is sent to the function when it is called.


##################################################
# 42. 如附圖所示為 Python 程式碼片段，其執行結果下列哪一項正確？

balance = 1000
def account_balance(deposit=0, withdrawal=0):
    global balance
    balance = 3000
    balance += deposit
    balance -= withdrawal
    print('balance: ', balance)

account_balance(600)
print('balance: ', balance)

# 答案： A

# 45. 下列哪一項「不」是 Spark 的執行模式？

# (A)	on job
# (B)	on yarn
# (C)	on cloud
# (D)	standalone

# 答案： A

# Spark四種執行模式
# Local 模式: 在1台電腦上執行 Spark.
# Standalone 模式: 在多台電腦, Master + Slave 所組成的 Spark 叢集執行, 須啟動 Master與Worker.
# Yarn 模式: Spark 用戶端可以直接連線 Yarn, 不需額外構立Spark叢集. 在 Hadoop架構中, 須啟動Yarn與HDFS.
# Mesos 模式: Spark用戶端直接連線 Mesos 叢集, 不需要額外構立 Spark 叢集.
# 參考資料: https://spark.apache.org/
    
"""
初級巨量資料分析師-111.11.19-11102-B12-科目2-資料處理與分析概論-疑義題釋義版.pdf
"""

##################################################
# 4. 下列哪一項是最「不」適合填補遺缺值（Missing Value）的方式？
# (A)	熱卡填補法（Hot Deck Imputation）
# (B)	迴歸填補法（Regression Imputation）
# (C)	填補最大值
# (D)	平均值填補（Mean/Mode Completer）

# 答案： C

# 熱卡填補法（Hot Deck Imputation）
# A once-common method of imputation was hot-deck imputation where a missing value was 
# imputed from a randomly selected similar record.
# https://en.wikipedia.org/wiki/Imputation_(statistics)

##################################################
# 12. 下列哪一種類型資料，適合使用資料增益（Information Gain, IG）進行特徵選取（Feature Selection）？
# (A)	擁有大量不同數值的資料特徵
# (B)	名目（Nominal）的資料特徵
# (C)	非離散化的數值特徵
# (D)	連續型的數值

# 答案： B

# 二元分類在決策樹(Decision Tree)學習過程中, 信息增益(Information Gain, IG)是特徵選擇的一個重要指標, 
# 它定義為一個特徵能夠為分類系統帶來多少信息, 帶來的信息越多, 說明該特徵越重要, 其信息增益也就越大。因此, 本題選(B).

##################################################
# 15. 關於特徵（屬性）萃取（Feature Extraction）與轉換（Transformation），下列敘述哪一項正確？
# (A)	資料縮減泛指屬性挑選（Selection）與萃取（Extraction）
# (B)	屬性越多，表示後續建模有越多參數要調校，過度配適（Overfitting）的風險越低
# (C)	各屬性的量綱均一化屬於屬性萃取（Extraction）的工作
# (D)	主成分分析（Principal Component Analysis, PCA）是分佈偏斜屬性常用的轉換方法

# 答案： A

# 處理偏斜資料的三大方法
import numpy as np
import pandas as pd
import seaborn as sns

# 混凝土強度估計資料集: https://github.com/rwepa/DataDemo#concretecsv
urls = 'https://raw.githubusercontent.com/rwepa/DataDemo/master/concrete.csv'
df = pd.read_csv(urls)
df = df[df['superplastic'] > 0]
df.head()
tmp = df.describe(include='all')

df['superplastic'].skew() # 1.77777

sns.histplot(data=df,
             x="superplastic", 
             kde=True,
             line_kws = {'linestyle':'dashed',
                         'linewidth':'1'}).lines[0].set_color('red')

# 1. Log Transform
superplastic_log = np.log(df['superplastic'])
superplastic_log.skew() # -0.391

# 2. Square Root Transform
superplastic_sqrt = np.sqrt(df['superplastic'])
superplastic_sqrt.skew() # 0.682

# 3. Box-Cox Transform
from scipy import stats
superplastic_boxcox = stats.boxcox(df['superplastic'])[0]
pd.Series(superplastic_boxcox).skew() # 0.034

##################################################
# 17. 若要確保巨量資料運算平台之服務，不會因為單點毀損導致無法存取服務
# （Single Point of Failure, SPOF），我們會進行高可用性（High Availability,
# HA）的設計，關於 HA 的敘述，下列哪一項錯誤？
# (A)	服務層級協議（Service-Level Agreement）決定連續不中斷服務的程度， 等級越高表示服務等級越高
# (B)	Hadoop 上的 HDFS（Hadoop Distributed File System）的高可用性可透過配置 Active/Active 兩個 NameNodes 節點解決 SPOF 問題
# (C)	可以透過  JournalNode  的設計來儲存  HDFS（ Hadoop Distributed File
# System）文件的紀錄，若發生 NameNode 損壞，新的 NameNode 可透過此紀錄恢復既有的文件紀錄
# (D)	具備有高可用性的架構下，發生 NameNode 損壞時，運行中的程式不受影響，仍會繼續完成工作

# 答案： B

# SPOF 問題(Single point of failure)
# Secondary namenode並不是HA的解法，它的作用是為了協助Namenode進行edits與fsimage的merge，
# 讓重啟的時候Namenood不會因為跑edits的檔案太多，以致於執行的時間很長。不過當namenode有問題的時候，
# 從secondary namenode載入metadata，也是可行的，但是還是要人工介入且也不是HA。
# https://ithelp.ithome.com.tw/articles/10133021

# high availability with Quorum Journal Manager (QJM) is preferred option.
# https://stackoverflow.com/questions/4502275/hadoop-namenode-single-point-of-failure

##################################################
# 20. 如附圖所示，有一個 data 數列，請問經過 MapReduce 模型處理的結果， 下列哪一項正確？
# data = [1,2,3,4,5]
# data.map(x ==> x * x).reduce(x, y => x + y)
# (A)	15
# (B)	55
# (C)	25
# (D)	49

# 答案： B

data = list(range(1,6))
data2 = data**2 # ERROR
data2 = [x**2 for x in data]
data2 # [1, 4, 9, 16, 25]
sum(data2) # 55

##################################################
# 22. 若隨機變數 X 服從二項分配（Binomial  Distribution），其每次試驗成功的機率為 0.4，
# 試驗次數為 10 次，則 X 的期望值與變異數，下列哪一項正確？
# (A)	期望值為 2 與變異數為 1.2
# (B)	期望值為 2 與變異數為 2.4
# (C)	期望值為 4 與變異數為 1.2
# (D)	期望值為 4 與變異數為 2.4

# 答案： D

# 考慮 b(n,p), E(X)=n*p, Var(X)=n*p*(1-p)
n = 10
p = 0.4
mean = n*p
print(mean) # 4
var = n*p*(1-p)
print(var) # 2.4

##################################################
# 24. 某公司所生產 10 公斤裝的糖果，其標準差為 0.4 公斤，欲估計母體平均數。
# 若 Z 為標準常態分配，且 Pr(Z < -1.96) = 2.5%，在 95%信賴水準下， 使估計誤差不超過 0.08 公斤，至少應抽多少包糖果來秤重？
# (A)	62 包
# (B)	75 包
# (C)	97 包
# (D)	116 包

# 答案： C

# 估計誤差 = [Z_[α/2]*σ]/sqrt(n)
# 估計誤差 <= 0.08
# 1.96*0.4/sqrt(n) <= 0.08
# n >= (1.96*0.4/0.08)**2 
(1.96*0.4/0.08)**2 # 96.04
# 因此, n=97.

##################################################
# 27. 某公司生產的燈泡有 10%的不良率。此公司為了品質對每一個燈泡做檢驗，將其分類為「通過」或「不通過」。
# 若檢驗員有 5%的機會分類錯誤， 則下列哪一項是被分類為「不通過」的百分比？
# (A) 12%
# (B) 14%
# (C) 16%
# (D) 18%

# 答案： B

# 10%*95%(分類正確)*90%*5%(分類錯誤)=14%

##################################################
# 32. 如附圖所示為 R 語言，執行 ggplot2 套件視覺化分析，下列敘述哪一項正確？

# 答案： D

# library(ggplot2)
# ggplot(mtcars, aes(hp,mpg)) +
#   geom_point() + 
#   geom_smooth(method = "lm")

##################################################
# 33. 若有兩個向量  A=<2,0,0>，B = <2,2,1>；請問這兩個向量之間的餘弦（Cosine）相似度最接近下列哪一個數字？
# (A) 0.67
# (B) 0.33
# (C) 1
# (D) 0.5

# 答案： A

# 餘弦相似度 (Cosine similarity) = cos(θ) = (A•B)/|A|•|B|
cos = (2*2 + 0*2 + 0*1)/((4**0.5)*(9**0.5))
print(cos) # 0.67

##################################################
# 47. 如附圖所示，關於二元分類（binary classification），若一分類模型產生之混淆矩陣（confusion matrix），該模型之精確度（precision）為下列哪一項？
# (A)  3 / 11
# (B)  8/ 20
# (C)  19 / 34
# (D)  8 / 11

# 答案： D

# http://rwepa.blogspot.com/2013/01/rocr-roc-curve.html
# Precision 精確度 = TP/(TP+FP) = 8 / (8+3)=8 / 11

##################################################
# 49. 如附圖所示，針對同一份資料建立的四種複迴歸模型，根據各種模型之指標資訊，請問下列哪一個為最佳模型？
# (A)	模型 1
# (B)	模型 2
# (C)	模型 3
# (D)	模型 4

# 答案： A

# Mallows's Cp: https://en.wikipedia.org/wiki/Mallows%27s_Cp
# 馬洛斯(Colin Lingwood Mallows)提出運用Cp去評估一個最小平方法(Ordinary Least Square或OLS)
# 為假設的線性迥歸模型的優良性, Cp數值越小模型準確性越高。同理, AIC, BIC 準則會選取數值較小者為較佳模型.
# end
