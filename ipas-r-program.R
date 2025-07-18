# File     : ipas-r-program.R
# Author   : Ming-Chang Lee
# Date     : 2011.07.04
# YouTube  : https://www.youtube.com/@alan9956
# RWEPA    : http://rwepa.blogspot.tw/
# GitHub   : https://github.com/rwepa
# Email    : alan9956@gmail.com

# reference: http://rwepa.blogspot.com/2013/01/r-201174.html
# 原檔名為 aMarvelousR_Lee(pp238).R, 用於 "A Marvelous R -  Foundation", 2023.4.3 更名為 ipas-r-program.R

# 更新所有套件
update.packages(ask = FALSE, checkBuilt = TRUE)

# Updated : 2020.06.03 更新程式碼
# Updated : 2021.04.11 新增 Chapter 6. iPAS-R-program (Chapter 6.iPAS - 科目二：資料處理與分析概論)
# Updated : 2021.06.01 新增 Chapter 7. ggplot2 套件
# Updated : 2021.08.27 新增 Chapter 8. 繪圖中文字型
# Updated : 2021.12.17 新增 Chapter 9. 長寬資料轉換(tidyr, reshape2套件)
# Updated : 2021.12.25 新增 Chapter 10.安裝專案套件
# Updated : 2022.05.04 新增 Chapter 11.匯入SAS檔案
# Updated : 2022.09.12 新增 Chapter 12.dplyr 套件
# Updated : 2024.08.02 新增 Chapter 13.正規表示式
# Updated : 2024.08.02 新增 Chapter 14.YouBike2.0臺北市公共自行車即時資訊資料分析
# Updated : 2024.09.04 新增 Chapter 15.RMarkdown輸出中文PDF錯誤
# Updated : 2025.07.18 新增 Chapter 16.JAVA_HOME設定

# 大綱 -----
# Chapter 0. R筆記
# Chapter 1. Basic R
# Chapter 2. Preparing Data
# Chapter 3. Graphics
# Chapter 4. Applied Statistics
# Chapter 5. Application
# Chapter 6. iPAS - 科目二：資料處理與分析概論
# Chapter 7. ggplot2 套件
# Chapter 8. 繪圖中文字型
# Chapter 9. 長寬資料轉換(tidyr, reshape2套件)
# Chapter 10.安裝專案套件
# Chapter 11.匯入SAS檔案
# Chapter 12.dplyr 套件
# Chapter 13.正規表示式
# Chapter 14.YouBike2.0臺北市公共自行車即時資訊資料分析
# Chapter 15.RMarkdown輸出中文PDF錯誤

# Chapter 0. R筆記 -----

# 2024.11.23
# fviz_nbclust {factoextra}: 估計集群法最佳集群個數與視覺化繪圖.

# Chapter 1. Basic R -----

# Change language settings
Sys.setenv(LANG = "en")

# p.10
# hist and plot
x <- c(1:100)
y <- rnorm(100)*100
hist(y)
test.model <- lm(y ~ x)
test.model
plot(x,y)
library(help="graphics")

# Understanding Diagnostic Plots for Linear Regression Analysis
# https://data.library.virginia.edu/diagnostic-plots/
# updated: 2022.9.27

# On-line help
?rnorm
?plot

#Information on package 'base'
library(help="base")
# end

# p.22
x <- c(1:10)
summary(x)
# end

# P.27
library(qcc)
?qcc

# p.28
# X-bar quality control chart
# load qcc library
library(qcc)
data(pistonrings)
attach(pistonrings)
pistonrings
diameter <- qcc.groups(diameter, sample)
qcc(diameter[1:25,], type="xbar")
# end

# TRY!
head(pistonrings)
head(pistonrings, n=3)
tail(pistonrings)
# end

# Chapter 2. Preparing Data -----

# p.40
# mode and length
x <- c(1:10)
mode(x)
length(x)
# end

# p.41
# name of object
A <- "WEPA"; compar <- TRUE; z <- 3+4i
mode(A); mode(compar); mode(z)
# end

# p.43
# Special numbers: Inf, -Inf, NaN, NA
x <- 5/0
x
exp(x)
exp(-x)
x - x
0/0
# NA
x.NA <- c(1,2,3)
x.NA
length(x.NA) <- 4
x.NA
# end

# p.46
# Joining (concatenating) vectors: c
x <- c(2,3,5,2,7,1)
x
y <- c(10,15,12)
y
z <- c(x, y)
z
# end

# p.47
# Subsets of vectors: Specify the numbers of the elements 
# that are to be extracted
# Assign to x the values 3, 11, 8, 15, 12
x <- c(3,11,8,15,12)

# Extract elements no. 2 and 4
x[c(2,4)]

# Use negative numbers to omit elements:
x[-c(2,4)]

# Generate a vector of logical (T or F)
x > 10

# Subset for user's defined conditions
x[x > 10]

# Vectors have named elements- method 1
c(ALAN=100, SERENA=2000, ANDY=300, ALPHA=400)[c("ALAN","ANDY")]

# Vectors have named elements- method 2
score <- c(ALAN=100, SERENA=2000, ANDY=300, ALPHA=400)
score[c("ALAN","ANDY")]
# end

# p.49
# Regular sequences: seq
x1 <- 1:100
x2 <- 100:1
x3 <- seq(1,10, 0.5)
x4 <- seq(length=9, from=1, to=5)
x5 <- c(1,2,2.5,6,10)
# Regular sequences: scan
x6 <- scan()

# p.50
# Regular sequences: rep
rep(1,5)

# Regular sequences: sequence
sequence(5)

# Error !!!
sequence(5,2)
# Now, it is correct for R-4.0.0.

# the concatenated sequences 1:5 and 1:2
sequence(c(5,2))

# the concatenated sequences 1:5, 1:2 and 1:3
sequence(c(5,2,3))
# end

# p.52
# gl
gl(3, 5)
gl(3, 5, length=30)
gl(2, 5, label=c("Male", "Female"))
# end

# TRY
x <- gl(3, 4, label=c("優良", "普通", "加油"), length=27)
x
# end

# p.54
# Built-in Constants
# Upper-case letters 
x <- LETTERS
x
length(x)

# Lower-case letters
y <- x[-c(2:10)]
y
length(y)

# Three-letter abbreviations for the month names
monthname.abb <- month.abb
monthname.abb
monthname.abb[c(1:10)]

# English names for the months 
monthname.full <- month.name
monthname.full

# Ratio of the circumference of a circle to its diameter
circle.area <- pi*10^2
circle.area
# end

# p.55
# NA
x <- c(pi, 1, 2,3)
x
x[c(2,4)] <- NA
x
is.na(x[2])
is.na(x[1])

# To replace all NAs by 0
x[is.na(x)] <- 0
x
# end

# p.58
# A vector of five numbers
v1 <- c(.29, .30, .15, .89, .12)
v1
class(v1)
typeof(v1)

# Coerces into a single type
v2 <- c(.29, .30, .15, .89, .12, "wepa")
v2
class(v2)
typeof(v2)

# vector(mode, length)
x1 <- vector(mode="numeric", length=1000000)

# View x1
head(x1)

# Verify a vector
is.vector(x1)

# vector
x2 <- c("Taiwan", "China", "USA")
x2
is.vector(x2)

# Expand the length of a vector
length(x2) <- 5
x2
# end

# p.61
# list
list.test1 <- list(1,2,3,4,5)
list.test1[1]
list.test1[[1]]

# Each element in a list may be given a name
product <- list(destination="Taipei",
                dimensions=c(3,6,10),price=712.19)
product[2]
product[[2]]
product$price
# end

# list all available data sets
data()
head(cars,n=3)
pts <- list(x=cars[,1], y=cars[,2])
plot(pts)
# end

# p.64
# factor
f1 <- factor(1:3)
f2 <- factor(1:3, levels=1:5)
f3 <- factor(1:3, labels=c("A", "B", "C"))
f4 <- factor(letters[1:6], label="YDU")
f4
class(f4)
eye.colors <- factor(c("brown", "blue", "blue", "green", 
                       "brown", "brown", "brown"))
eye.colors
levels(eye.colors)
# end

# p.66
# array
a1 <- array(letters)
class(a1)
dim(a1)

# array
a2 <- array(1:3, c(2,4))
a2
dim(a2)
length(a2)
a2[1, ] # select row 1
a2[, 4] # select column 4
# end

# p.67
# array
a3 <- array(data=1:24,dim=c(3,4,2))
a3
# end

# p.69
# matrix
matrix.data <- matrix(c(1,2,3,4,5,6), 
                      nrow = 2, ncol = 3, byrow=TRUE, 
                      dimnames = list(c("row1", "row2"), c("C1", "C2", "C3")))
matrix.data
# end

# p.71
# data.frame
x <- c(1:4); n <- 10; m <- c(10, 35); y <- c(2:4)
df1 <- data.frame(x, n)
df1
df2 <-data.frame(x, m)
df2
# end

# p.71
# TRY !
df3 <- data.frame(x, y) # ERROR
df4 <- data.frame(var1= rnorm(5), var2=LETTERS[1:5])
df4
# end

# p.72
# The data give the speed of cars and 
# the distances taken to stop.
data(cars)

help(cars)

class(cars)

head(cars)
# end

# p.75
# Time-series
data.ts <- ts(c(2,5,4,6,3,7,9:8),start=c(2009,2),frequency=4)
data.ts
is.ts(data.ts)
start(data.ts)
end(data.ts)
frequency(data.ts)
deltat(data.ts) # 0.25(=1/4)
plot(data.ts, type="b")
# end

# p.79
# Accessing Data
x <- c(1:8)

# how many elements?
length(x)

# ith element, i=2
x[2]

# all but ith element, i=2
x[-2]

# first k elements, k=5
x[1:5]

# last k elements, k=5
x[(length(x)-5+1):length(x)]

# specific elements
x[c(1,3,5)]

# all greater than some value
x[x>3]

# biger than or less than some values
x[x<3 | x>7]

# which indices are largest
which(x==max(x))
# end

# p.82
# Import/Export data
# Create a data directory C:\R.data
# Get working directory
getwd()

# Set working directory
dir.create("C:/R.data")
workpath <- "C:/R.data"
setwd(workpath)
getwd()

# p.86
# Create a CSV file (C:\R.data\score.csv)
# in which each field is separated by commas.
# Import dataset

# https://github.com/rwepa/DataDemo/blob/master/score.csv

score1 <- read.table(file="score.csv", header= TRUE, sep=",")

# TRY !
# score1 <- read.table(file="score.csv", header= TRUE)
score1
dim(score1)
names(score1)
row.names(score1)


# Add new column data for mid_term
mid.term <- matrix(c(60,80,65,85,80,90,99), nrow=7, ncol=1, 
                   byrow=FALSE,   dimnames = list(c(),c("mid.term")))
mid.term

# Merge two data.frame( score1 and mid_term)
score2 <- data.frame(score1, mid.term)
score2

# Export dataset
write.table(score2 , file= "score.final.txt", 
            sep = "\t", 
            append=FALSE, 
            row.names=FALSE, 
            col.names = TRUE, 
            quote= FALSE)
# end

# Chapter 3. Graphics -----

# p.91
demo(graphics)
# end

# p.96
# plot
data(cars)
head(cars, n=3)
plot(cars) # x-axis:speed; y-axis: dist

# p.97
# use the variable symbol "$"
plot(cars$dist, cars$speed)

# p.98
plot(cars, type="b")
# end

# p.99
# Available colors (#657)
# colors()
plot(cars, type="b", pch=5, col="red",
     xlab="Speed(mph)", ylab="Stop distance(ft)", 
     main="Speed and Stopping Distances of Cars",
     sub= "Figure 1: Plotting demonstration")
# end

# p.100
# Barplot
CarArrived <- table(NumberOfCar <- rpois(100, lambda=5))
CarArrived
barplot(CarArrived)
barplot(CarArrived, col=rainbow(14))
#end

# p.101
# pie
# Sales ratio
pie.sales <- c(0.14, 0.30, 0.26, 0.15, 0.10, 0.05)
# Sales area
names(pie.sales) <- c("Taipei1", "Taipei2", "Taipei3", "Taichung", 
                      "Kao", "Other")
# default colours
pie(pie.sales)
# end

# p.102
# pie with customized colour
pie(pie.sales, col = c("purple", "violetred1", "green3", 
                       "cornsilk", "cyan", "white"))
# end

# p.103
# pie with density of shading lines
pie(pie.sales, density = c(5,20), clockwise=TRUE)
# end

# p.104
# Box-and-whisker plot(s) of the given (grouped) values.
mat <- cbind(Uni05 = (1:100)/21, Norm=rnorm(100), T5 = rt(100,df= 5), 
             Gam2 = rgamma(100, shape = 2))
head(mat)
boxplot(data.frame(mat), main = "boxplot")
# end

# p.106
# stem plot
mat <- round(rnorm(10,30,10),0)
mat
stem(mat)
# end

# p.108
# Customized plot for 6-Sigma Quality Level
z <- pretty(c(44,56), 50) # Find 50 equally spaced points
ht <- dnorm(z, mean=50,sd=1) # By default: mean=0, standard deviation=1
plot(z, ht, type="l", main="6-Sigma Quality Level", 
     xlab="Qualty characteristic", ylab="Quality Level" , 
     axes=FALSE, xlim=c(42,58), ylim=c(0,0.5))

# Add axis
# 1=below, 2=left, 3=above and 4=right
axis(side=1, c(42:58),tick = TRUE)
axis(side=2, tick = TRUE)

# Add vertical line
# h=0: horizontal line(s);v=0: vertical line(s)
abline(v=c(44,50,56), lty=c(1,2,1), col=c("red","blue","red"))  

# Add text
text(44,0.5,"LSL", adj = c(-0.2,0))
text(50,0.5,"T",adj = c(-0.2,0))
text(56,0.5,"USL",adj = c(-0.2,0))
# end

# p.111
# 3D plot
# scatterplot3d package
# setup plotting environment for 2 rows and 2 columns
# par(bg = "white")
op <- par(mfrow = c(2,2))

# method 1: scatterplot3d (type=point) - Parabola
library(scatterplot3d)
x <- seq(-3, 3, length = 30)
y <- x
f <- function (x,y) {a <- 9; a - x^2 - y^2}
scatterplot3d(x, y, f(x,y),
              highlight.3d=TRUE, col.axis="blue",
              pch=20,
              main="Euclidean Utility Function - parabola, (type=point)",
              xlab="X", ylab="Y", zlab="f(x,y)", 
              zlim=c(0,9),
              col.grid="lightblue", 
              type="p"
)

# method 2: scatterplot3d (type=point)
library(scatterplot3d)
x <- seq(-3, 3, length = 30)
f <- function (x,y) {a <- 9; a - x^2 - y^2}
x1 <- rep(x, 30)
x2 <- rep(x, each=30)
znew <- f(x1, x2)
scatterplot3d(x1, x2, znew,
              highlight.3d=TRUE, col.axis="blue",
              pch=20,
              main="Euclidean Utility Function, (type=point)",
              xlab="X", ylab="Y", zlab="f(x,y)", 
              zlim=c(0,9),
              col.grid="lightblue", 
              type="p"
)

# method 3: scatterplot3d (type=line)
library(scatterplot3d)
x <- seq(-3, 3, length = 30)
f <- function (x,y) {a <- 9; a - x^2 - y^2}
x1 <- rep(x, 30)
x2 <- rep(x, each=30)
znew <- f(x1, x2)
scatterplot3d(x1, x2, znew,
              highlight.3d=TRUE, col.axis="blue",
              pch=20,
              main="Euclidean Utility Function (type=line)",
              xlab="X", ylab="Y", zlab="f(x,y)", 
              zlim=c(0,9),
              col.grid="lightblue", 
              type="l"
)

# method 4: persp
x <- seq(-3,3,length = 30)
y <- x
f <- function (x,y) { a <- 9; a-x^2-y^2}
z <- outer(x,y,f)
persp(x,y,z,zlim = range(c(-10:10), na.rm = TRUE), 
      expand=1,theta = 30, phi = 30,
      col = "lightblue",ticktype="detailed", 
      xlab="X", ylab="Y", zlab="f(x,y)",
      main="Euclidean Utility Function")

# method 5: misc3d (need "rgl" package)
# misc3d package
library(misc3d)
parametric3d(
  fx = function(u, v) u,
  fy = function(u, v) v,
  fz = function(u, v) -9 - u^2 - v^2 ,
  fill = FALSE,
  color = "blue",
  scale = FALSE,
  umin = -3, umax = 3, vmin = -3, vmax = 3, n = 100)

# setup plotting environment to the default
# par(mfrow=c(1,1))
par(op)
# end

# p.120
# 3D plot - sample
?persp
demo(persp)
# end

# Chapter 4.Applied Statistics -----

# p.122
# descriptive statistics
# Set working directory
workpath <- "C:/R.data"
setwd(workpath)
# import data
score <- read.table(file="score.csv", header= TRUE, sep=",")
# view data
score
# end

# summary data
# TRY summary(score)
score[2]
score[, 2]
score[3, ]
score$quiz1
quiz1 <- score$quiz1
mean(quiz1)
max(quiz1)
min(quiz1)
# end

# standard deviation
std(quiz1) # error function

# solution 1
sqrt( sum( (quiz1 - mean(quiz1))^2 /(length(quiz1)-1)))

# solution 2
# user's function
std = function(x) sqrt(var(x))
std(quiz1)

# solution 3
sd(quiz1)
# end

# p.129
# sample- t.test
# H0: u=95, H1: u<>95
t.test.data <- rnorm(50, mean=100, sd=15)
t.test(t.test.data, mu=95)
# p value is small, reject H0
# end

# p.130
# wilcox.test(x, conf.int=TRUE)
wilcox.test.data <- rnorm(50, mean=100, sd=15)
wilcox.test(wilcox.test.data, conf.int=TRUE, conf.level=0.99)
# end

# p.133
# prop.test
prop.test(39,215,0.15)
# end

# p.135
# shapiro.test
shapiro.test(rnorm(100, mean = 5, sd = 3))
shapiro.test(runif(100, min = 2, max = 4))
# end

# p.139
# QQ plot without logarithmic transformation
# first: qqnorm, second: qqline
data(Cars93, package="MASS")
qqnorm(Cars93$Price, main="Q-Q Plot: Price")
qqline(Cars93$Price)

# QQ plot without logarithmic transformation
qqnorm(log(Cars93$Price), main="Q-Q Plot: log(Price)")
qqline(log(Cars93$Price), col=2)
# end

# p.145
# one-way anova
# Copy the data to C:\R.data

# Import data
# https://github.com/rwepa/DataDemo/blob/master/drink.csv

drink.sales <- read.table("drink.csv", header=TRUE, sep=",")
head(drink.sales)

# drink.type <- factor(gl(4,5,label=c(letters[1:4])))
drink.type <- gl(4,5,label=c(letters[1:4]))
drink.type

drink <- data.frame(drink.type=drink.type, drink.sales)
head(drink)
class(drink)

# method 1. oneway.test
drink.oneway <- oneway.test(drink$sales ~ drink$drink.type, 
                            var.equal=TRUE)
drink.oneway

# method 2. aov
drink.anova <- aov(drink$sales ~ drink$drink.type)
summary(drink.anova)

# method 3. Linear model
drink.lm <- lm(drink$sales ~ drink$drink.type)
anova(drink.lm)
# end

# p.149
# linear regression
# Build-in data: cars
# x: speed, y: dist
head(cars)
dim(cars)

# linear model
cars.lm <- lm(dist~speed, data=cars)
summary(cars.lm)
# end

# regression information
anova(cars.lm)
coefficients(cars.lm)
coef(cars.lm)
confint(cars.lm)
deviance(cars.lm)
effects(cars.lm)
fitted(cars.lm)
residuals(cars.lm)
resid(cars.lm)
summary(cars.lm)
vcov(cars.lm)
# end

# residual plot
plot(dist ~ speed, 
     data = cars, 
     xlab = 'Speed', 
     ylab = 'Stopping distance', 
     main = 'Speed vs. Stopping distance for cars')

abline(cars.lm)
points(cars$speed, fitted(cars.lm), pch=18, col = "blue")
segments(cars$speed, cars$dist, cars$speed, fitted(cars.lm), col = "green")

# Chapter 5.Application -----

# p.157
# Rcmdr package
# install.packages("Rcmdr", dependencies=TRUE)

# p.160
# Run in native R, NOT FOR RSTUDIO.
library(Rcmdr)

# p.169
# View all available data sets
data()
# end

# p.171
# List the data sets in specific package 
data(package="qcc")
# end

# p.174
# Load the data sets in specific package
data(pistonrings, package="qcc")
# end

# p.174
# TRY
# Select data with "sampe=1"?
pistonrings.sample1 <- pistonrings[pistonrings$sample == 1,]
pistonrings.sample1
# end

# P.177
# Help and column names
help("pistonrings")
names(pistonrings)
# end

# p.182
# set global options
x <- sqrt(2)
x
options(digits=16)
x
# end

# P.183
# 下載10萬筆測試資料
# https://github.com/rwepa/DataDemo/blob/master/R03_Orders.txt

# p.193
# Probability function 
dnorm(1.96, 0, 1)
pnorm(1.96, 0, 1)  # z --> p, 0.975
qnorm(0.975, 0, 1) # p --> z, 1.96
rnorm(5, 0, 1)
# T分配: pt, F分配: pf, chi-squared分配: pchisq, 指數分配: pexp, gamma分配: pgamma, Weibull分配: pweibull

dnorm(1.645)
pnorm(1.645)
pnorm(1.96)
pnorm(2)
qnorm(0.95, 0, 1)
# end

# p.194
# Random generation
# also runif(1,min=0,max=2)
runif(1,0,2) 

runif(5,0,2)

# 5 random numbers in [0,1]
runif(5) 

x <- runif(100) # get the random numbers U(0,1)
hist(x,probability=TRUE,col=gray(.9),main="uniform on [0,1]")
# end

# p.195
# Sample - binomial distribution
pbinom(3, 10, 0.1)
# end

# p.197
# Sample - normal distribution
pnorm(c(2.5), mean=0, sd=1, lower.tail=TRUE)
# end

# p.197
# The ppt is disabled.
# TRY P(Z<=a)=0.95
a <- qnorm(c(0.95), mean=0, sd=1, lower.tail=TRUE)
a
# end

# p.219
# Quality Control Chart
library(qcc)
workpath = "c:/rdata"
setwd(workpath)

# 先將資料複製到 C:\rdata 目錄
# 匯入資料

# https://github.com/rwepa/DataDemo/blob/master/hw1.csv
hw1 <- read.table("hw1.csv", header=TRUE, sep=",")
# 顯示資料
hw1

# 取出全部資料的第2-4欄
pcb <- hw1[, c(2:4)]; pcb

dim(pcb) # 包括 25列, 3行 資料

qcc(pcb, type="xbar") # 第22筆資料超出管制界限

qcc(pcb, type="R")    # 第15筆資料超出管制界限

pcb.modified <- pcb[-c(15,22), ] # 移除第15,22筆資料

dim(pcb.modified) # 包括 23列, 3行 資料

qcc(pcb.modified, type="xbar") # 製程有在管制界限之內嗎?

qcc(pcb.modified, type="R") # 製程有在管制界限之內嗎?
# end

# p.233
# The example of R for Support Vector Machines(SVM)
# 安裝SVM 套件 e1071
# 載入套件 e1071
library(e1071)
library(mlbench)

# 載入資料集 Glass in mlbench package
# 資料集 214 個觀測值,9 個變數,第9 個數數名稱為Type,
# 有7 個種類(1:7)
data(Glass)
head(Glass)

# 設定變數index 為編號1,2,…214.
index <- 1:nrow(Glass)

# 準備隨機抽樣並設定測試資料的編號
# 利用sample 取樣,將資料的1/3 做為測試資料的序號
testindex <- sample(index, trunc(length(index)/3))
# 設定測試資料 testset,共71 筆資料
testset <- Glass[testindex, ]
# > dim(testset) # 可知道有71 筆測試資料
# 將其他資料設定訓練資料 trainset,共143 個筆資料
trainset <- Glass[-testindex, ]

# 利用svm 執行並將結果存入變數svm.model
svm.model <- svm(Type ~ ., data = trainset, cost = 100, gamma = 1)

# 利用predict 執行測試資料的分類預測
svm.pred <- predict(svm.model, testset[, -10])
print(svm.pred)

# 利用 write 輸出結果
# 將結果輸出成CSV 檔案
write.table(svm.pred, file = "svm.test.csv", sep = ",")
# end

# Chapter 6.iPAS - 科目二：資料處理與分析概論 -----

setwd("C:/rdata")

# 1-1資料組織與清理 -----

# KNN (K近鄰法) demo -----
library(animation)

# 設定動畫參數
ani.options(interval = 1, nmax = 10)

# 建立訓練集,測試集
set.seed(168)
df <- iris[iris$Species != "setosa",]
df$Species <- factor(df$Species)
ind <- sample(2, nrow(df), replace = TRUE, prob = c(0.8, 0.2))
traindata <- df[ind == 1, 3:4]
testdata <- df[ind == 2, 3:4]

# KNN示範
knn.ani(train = traindata, test = testdata, cl = df$Species[ind == 1], k = 20)

# Kmeans (集群法) demo -----
library(animation)
kmeans.ani()

# 資料標準化 -----
data(Cars93, package = "MASS")

# 直方圖
hist(Cars93$Price)
summary(Cars93$Price) # 原始資料 [7.4, 61.9]

# (0,1)標準化 -----

PriceMin <- min(Cars93$Price)
PriceMax <- max(Cars93$Price)

Cars93$PriceZeroOne <- (Cars93$Price - PriceMin)/(PriceMax - PriceMin)
head(Cars93$PriceZeroOne)
summary(Cars93$PriceZeroOne) # (0,1)標準化 [0, 1]

# min-max 標準化 -----

PriceMinNew <- 1
PriceMaxNew <- 10

Cars93$PriceMinMax <- PriceMinNew + 
  ((Cars93$Price - PriceMin)/(PriceMax - PriceMin))*(PriceMaxNew - PriceMinNew)

summary(Cars93$PriceMinMax)

# 標籤編碼 (Label encoding) -----

# 範例1 german_credit
# 參考 <<<R商業預測與應用>>>第3章 監督式學習商業預測
# https://courses.mastertalks.tw/courses/R-2-teacher

# https://github.com/rwepa/DataDemo/blob/master/german_credit.csv
credit <- read.csv("german_credit.csv") # 1000*10

str(credit)

head(credit)

# label encoding -----
credit$RiskEncoding <- ifelse(credit$Risk == "good", 1, 0)

head(credit$RiskEncoding)

table(credit$RiskEncoding)

# One-hot encoding -----

# Job 工作 {0,1,2,3}
# 0 - unskilled and non-resident 非技術人員和非居民
# 1 - unskilled and resident 非技術人員和居民
# 2 - skilled 技術人員
# 3 - highly skilled 高度技術人員

# 轉換為 factor
credit$JobOneHot <- factor(credit$Job, label = c("unskilled and non-resident", 
                                                 "unskilled and resident", 
                                                 "skilled", 
                                                 "highly skilled"))

str(credit$JobOneHot)

levels(credit$JobOneHot)

# method 1-使用 model.matrix {stats}
myonehot <- model.matrix(object = ~ JobOneHot - 1, data = credit) # matrix
head(myonehot)

# method 2-使用 dummyVars {caret}
library(caret)

dummy <- dummyVars(" ~ JobOneHot", data = credit)
dummy

# 範例2 鐵達尼號 : 使用 dummyVars 直接進行預測
library(earth)

data(etitanic, package = "earth")

head(etitanic)

head(model.matrix(survived ~ ., data = etitanic))

dummies <- dummyVars(survived ~ ., data = etitanic)

etitanic$PedictSurvived <- predict(dummies, newdata = etitanic)

head(etitanic)

# 1-2.資料摘要與彙總 -----

# 盒鬚圖 boxplot -----
data(Cars93, package = "MASS")

boxplot(Cars93$Price)

# 盒鬚圖的5個指標
# Lower bound, 25% quantile, Median, 75% quantile, Upper bound
# 下邊界, 25百分位數, 中位數, 75百分位數, 上邊界
Cars93_Price <- boxplot(Cars93$Price)
Cars93_Price

# 群組盒鬚圖 - 基礎繪圖
boxplot(Price ~ Origin, data = Cars93)

# 盒鬚圖 - ggplot2 -----

library(ggplot2)
p <- ggplot(Cars93, aes(y = Price)) + 
  geom_boxplot()
p

# 群組盒鬚圖 - ggplot2
p1 <- ggplot(Cars93, aes(x = Origin,y = Price)) + 
  geom_boxplot()
p1

# 匯出資料
write.table(Cars93, file = "C:/rdata/Cars93.csv", sep =",", row.names = FALSE)

# R - 排序 sort, order -----

x <- c(9,2,6,3,1)

sort(x)

order(x)

x[order(x)]

sort(x, decreasing = TRUE)

# 資料框排序 -----

df <- head(iris, n = 5)

# 遞增排序
df[order(df$Sepal.Length),]

# 遞減排序
df[order(df$Sepal.Length, decreasing = TRUE),]

# 群組個數 table -----
data(Cars93, package = "MASS")

table(Cars93$AirBags)

# 群組個數 table-2個維度

table(Cars93$AirBags, Cars93$Origin)

# 群組邊界計算 addmargins-預設值為總和
addmargins(table(Cars93$AirBags, Cars93$Origin))

# 群組邊界計算 addmargins-mean
addmargins(table(Cars93$AirBags, Cars93$Origin), FUN = mean)

# 群組百分比計算 prop.table
prop.table(table(Cars93$AirBags, Cars93$Origin))

# table 多維度: 安全氣囊, 進口別, 傳動系統
table(Cars93$AirBags, Cars93$Origin, Cars93$DriveTrain)

# 類別平均值計算
aggregate(formula = Price ~ AirBags, data = Cars93, FUN = mean)

aggregate(formula = Price ~ AirBags + Origin, data = Cars93, FUN = mean)

# 摘要 -----
summary(Cars93)

# 1-3.屬性轉換與萃取 -----

# 奇異值分解 (Singular Value Decomposition, SVD) -----
x <- matrix(1:6, byrow = TRUE, ncol = 2)
x
(s <- svd(x))
class(s) # list

# Mean normalization -----
# (x - mu)/(max - min)

x <- c(1, 2, 2, 4, 5)
(xmean <- mean(x))
(xmax <- max(x))
(xmin <- min(x))
(x - xmean)/(xmax - xmin)

# 裝箱 (Binning) -----

?cut

data(Cars93, package = "MASS")

quantile(Cars93$Price)

Cars93$Price

cut(x = Cars93$Price, breaks = c(0, 13, 23, Inf), labels = c("低", "中", "高"))

# scale
x <- head(Cars93[c('Price', 'Horsepower')])
x

apply(x, 2, mean)
apply(x, 2, sd)
scale(x)

op <- par(mfrow=c(1,2))
hist(Cars93$Price)
hist(scale(Cars93$Price))
par(op)

# 2-1.統計分析基礎 -----

# 百分位數 -----
# 預設值 0% 25% 50% 75% 100%
quantile(iris$Sepal.Length)

# 平均數𝜇之區間估計 - 母體變異數已知 -----
# 95%之信賴區間
alpha <- 0.05
sampleSD <- 5
sampleSize <- 16
sampleMean <- 60
zScore <- qnorm(p = alpha/2,  lower.tail = FALSE)
zScore

lowerBound <- sampleMean - zScore*sampleSD/sqrt(sampleSize)
upperBound <- sampleMean + zScore*sampleSD/sqrt(sampleSize)
print(c(lowerBound, upperBound))
# [1] 57.55005 62.44995

# 99%之信賴區間
alpha <- 0.01
zScore <- qnorm(p = alpha/2,  lower.tail = FALSE)
zScore

lowerBound <- sampleMean - zScore*sampleSD/sqrt(sampleSize)
upperBound <- sampleMean + zScore*sampleSD/sqrt(sampleSize)
print(c(lowerBound, upperBound))
# [1] 56.78021 63.21979

# t-檢定 -----
set.seed(168)
x <- rnorm(n = 10, mean = 5)
x           
t.test(x, mu = 5)

# 卡方檢定 -----
chisq.test(c(230,220,450))

mytest <- chisq.test(c(230,220,450))
names(mytest)
mytest$p.value

# 2-2.探索式資料分析與非監督式學習 -----

# 使用 dbscan {fpc} -----

library(fpc)
# https://cran.r-project.org/web/packages/fpc/index.html

df <- iris[,-5]

df.dbscan <- dbscan(df, eps=0.42, MinPts=5)

df.dbscan
# 資料分成3群(編碼1,2,3)
# 編碼0(第1行)表示離群值/噪音值

# border 邊界點
# seed   密度點
# DBSCN  計算集群的平均值(中心點)與K-means意義相同嗎?

# 2-3.線性模型與監督式學習 -----

# 線性模型 – R: lm {stats}
# 參考 https://github.com/rwepa/DataDemo/blob/master/marketing.R

# 下載資料
# https://github.com/rwepa/DataDemo/blob/master/marketing.csv

# 建立 y = β0 + β1 * x1 + β2 * x2 + β3 * x3

# x1: 自變數 youtube, x2: 自變數 facebook, x3: 自變數 newspaper
# y: 依變數 sales

sales_lm_all <- lm(sales ~ youtube + facebook + newspaper, data = marketing)

# 檢視模型結果
summary(sales_lm_all)

# 結果包括4大項目
# 1.Call: lm() 線性
# 2.Residuals: 線性迴歸模型的殘差
# 3.Coefficients: 迴歸係數, newspaper 的p值 大於 0.05, 考慮刪除此自變數.
# 4.統計值: 殘差標準差, R平方, 調整後R平方, F統計值, 自由度(DF), p-value

# Chapter 7. ggplot2 套件 -----

# https://cran.r-project.org/web/packages/ggplot2/index.html
# ggplot2: Create Elegant Data Visualisations Using the Grammar of Graphics
# 參考資料 https://ggplot2-book.org/

# ggplot2概念
# 1.以圖層(layers)為基礎的繪圖套件,它實現了Wilkinson (2005)的繪圖文法(Grammar of Graphics)概念.
# 2.一個圖形是由數個圖層所組成,其中一層包含了資料(data)層.
# 3.Wilkinson認為圖形繪製須結合數據與繪製規範,規範並非是圖形視覺效果的名稱(例如:長條圖,散佈圖,直方圖等).
# 4.規範應是一組共同決定圖形如何建立的規則 – a grammar of graphics.

library(ggplot2)

?diamonds

str(diamonds) # 53940*10

set.seed(168)

dsmall <- diamonds[sample(nrow(diamonds), 500),]

# ggplot: 散佈圖
p <- ggplot(data=dsmall, mapping=aes(carat, price, color=color)) + 
  geom_point(size=4)
p

# ggplot: 散點圖+線性迴歸
p <- ggplot(dsmall, aes(carat, price)) + 
  geom_point() +
  geom_smooth(method="lm", se=FALSE)
p

p <- ggplot(dsmall, aes(carat, price)) + 
  geom_point() +
  geom_smooth(method="lm", se=TRUE)
p

# ggplot: 散點圖+群組線性迴歸
ggplot(dsmall, aes(carat, price, group=color)) + 
  geom_point(aes(color=color), size=2) + 
  geom_smooth(aes(color=color), method="lm", se=FALSE)

# ggplot: 線圖
p <- ggplot(iris, aes(Petal.Length, Petal.Width, group=Species, color=Species)) +
  geom_line()
p

# ggplot: 線圖+分面 facet_wrap()
p <- ggplot(iris, aes(Sepal.Length, Sepal.Width)) + 
  geom_line(aes(color=Species), size=1) + 
  facet_wrap(~Species, ncol=1)
p

# ggplot: facet_wrap 分面-1維
ggplot(mpg, aes(displ, hwy)) + 
  geom_point() + 
  facet_wrap(~class)

# ggplot: facet_grid 分面-2維, (列, 行)
ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  facet_grid(cols = vars(cyl), rows = vars(class))

# ggplot2: 線圖+散佈圖
ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  geom_line()

# ggplot2: 散佈圖+線圖+群組
ggplot(mpg, aes(displ, hwy)) +
  geom_point(aes(color=factor(cyl))) +
  geom_line()

# R軟體開放資料應用-高速公路篇 使用ggplot2
# http://rwepa.blogspot.com/2019/05/highway.html

# Chapter 8. 繪圖中文字型 -----

# shiny 繪圖 plot 中文字型錯誤 -----

# 方法1 使用 family 參數
# Example 1. 使用 Windows 微軟正黑體字型
# plot(..., family = "Microsoft JhengHei UI")

# 方法2 使用 showtext 套件
# Example 2. shiny app
library(shiny)
library(showtext)

## Loading Google fonts (https://fonts.google.com/)
font_add_google(name = "Noto Sans TC", family = "twn")
showtext_auto()
hist(..., family = "twn")

# Example 3. shiny app
library(shiny)
library(showtext)
showtext_auto()
ui <- fluidPage(
  titlePanel("iris散佈圖矩陣-2021.8.27"),
  mainPanel(
    plotOutput("distPlot")
  )
)
server <- function(input, output) {
  output$distPlot <- renderPlot({
    pairs(iris[-5], pch=16, col=iris$Species, 
          main = "iris散佈圖矩陣",
          cex.main=2)
  })
}
shinyApp(ui = ui, server = server)

# Chapter 9. 長寬資料轉換(tidyr, reshape2套件) -----

# 長寬資料轉換 long and wide data -----
olddata_wide <- read.table(header=TRUE, text="
                           subject sex control cond1 cond2
                           1   M     7.9  12.3  10.7
                           2   F     6.3  10.6  11.1
                           3   F     9.5  13.1  13.8
                           4   M    11.5  13.4  12.9
                           ")
# subject 欄位轉換為 factor
olddata_wide$subject <- factor(olddata_wide$subject)
str(olddata_wide)
olddata_wide

olddata_long <- read.table(header=TRUE, text='
                           subject sex condition measurement
                           1   M   control         7.9
                           1   M     cond1        12.3
                           1   M     cond2        10.7
                           2   F   control         6.3
                           2   F     cond1        10.6
                           2   F     cond2        11.1
                           3   F   control         9.5
                           3   F     cond1        13.1
                           3   F     cond2        13.8
                           4   M   control        11.5
                           4   M     cond1        13.4
                           4   M     cond2        12.9
                           ')
# subject 欄位轉換為 factor
olddata_long$subject <- factor(olddata_long$subject)
str(olddata_long)
olddata_long

# tidyr 套件
library(tidyr)

# gather: From wide to long
data_long <- gather(olddata_wide, condition, measurement, control:cond2)
data_long

# spread: From long to wide
data_wide <- spread(olddata_long, condition, measurement)
data_wide

# reshape2 套件
library(reshape2)

# melt: From wide to long

# Specify id.vars: the variables to keep but not split apart on
# method 1
melt(olddata_wide, id.vars=c("subject", "sex"))

# method 2
data_long <- melt(olddata_wide,
                  # ID variables - all the variables to keep but not split apart on
                  id.vars=c("subject", "sex"),
                  # The source columns
                  measure.vars=c("control", "cond1", "cond2" ),
                  # Name of the destination column that will identify the original
                  # column that the measurement came from
                  variable.name="condition",
                  value.name="measurement")
data_long

# dcast: From long to wide
data_wide <- dcast(olddata_long, subject + sex ~ condition, value.var="measurement")
data_wide

# Chapter 10.安裝專案套件 -----

# 考慮某專案須使用 ggplot2, tidyr, reshape2 套件,使用客製化函數 verifyPackage,執行套件之檢視,如果系統沒有安裝套件,則自動安裝該套件.

needPackage <- c("ggplot2", "tidyr", "reshape2")

verifyPackage <- function(needPackage) {
  for (x in needPackage) {
    if (!x %in% installed.packages()[,"Package"])
      install.packages(x)
  }
}
verifyPackage(needPackage)

# Chapter 11.匯入SAS檔案 -----

library(sas7bdat)

# 模擬資料-全民健保處方及治療明細檔_西醫住院
# https://github.com/rwepa/DataDemo/blob/master/h_nhi_ipdte103.sas7bdat

filname <- "h_nhi_ipdte103.sas7bdat"
system.time(dd2014 <- read.sas7bdat(filname))
dd2014

# 2023.5.17 updated
# 載入套件
library(haven)

# 匯入資料
dd2014 <- read_sas("h_nhi_ipdte103.sas7bdat")

# 執行系統時間
system.time(dd2014 <- read_sas("h_nhi_ipdte103.sas7bdat")) # 0.43秒
#   user  system elapsed 
#   0.21    0.00    0.43 
 
# 顯示資料
dd2014 # 14,297 × 80

# 類別
class(dd2014) # "tbl_df" "tbl" "data.frame"

# Chapter 12.dplyr 套件 -----

# r cran dplyr
# https://cran.r-project.org/web/packages/dplyr/index.html
# dplyr: A Grammar of Data Manipulation 資料操作文法
# dplyr = data frame + plyr

# plyr 發音 plier (鉗子), plyr 是 dplyr 的前個套件版本, 作者推薦使用 dplyr 套件
# plyr: https://www.slideshare.net/hadley/plyr-one-data-analytic-strategy

# filter                 : 條件式篩選資料
# slice                  : 列的指標篩選資料
# arrange                : 排序
# select                 : 選取行/更改欄位名稱
# rename                 : 選取所有行/更改欄位名稱
# distinct               : 選取不重覆資料
# mutate                 : 新增欄位,保留原資料
# transmute              : 新增欄位,不保留原資料
# summarise              : 群組計算

library(dplyr)

library(nycflights13) # 2013年NYC機場航班資料, 33萬筆資料 -----

flights # 336776*19

class(flights) # "tbl_df" "tbl" "data.frame"

# 如何轉換為 tbl_df, 使用 as.tbl -----
mytbl <- as.tbl(iris) # deprecated in dplyr 1.0.0.
mytbl <- tibble::as_tibble(iris)
class(mytbl)

# 資料結構與摘要 -----
str(flights)

summary(flights) # 有NA

head(flights)

tail(flights) # 注意:資料不是依照月,日排序

# filter 條件式篩選資料 -----
filter(flights, month == 1, day == 1)

flights[flights$month == 1 & flights$day == 1, ] # 基本指令, same as filter

filter(flights, month == 1 | month == 2) # AND 條件篩選, 同理 OR 使用 | 

# slice 列的指標篩選資料 -----
slice(flights, 1)           # 第1筆

slice(flights, n())         # 最後一筆

slice(flights, 123456:n())  # 第123456筆至最後一筆

# arrange 排序 -----
arrange(flights, year, month, day) # 依照年,月,日遞增排序

arrange(flights, desc(arr_delay)) # 依照延誤時間遞減排序

# select 選取行  -----
select(flights, year, month, day)

# Select 選取行, 使用 : -----
select(flights, year:dep_delay)

# select 選取行, 使用 負號, 表示刪除 -----
select(flights, -(year:day))

# select 選取行並且更改欄位名稱 -----
select(flights, tail_num = tailnum) # select 只選取 tailnum 1行資料

# select 選取行, 使用 starts_with 等參數
iris %>% select(starts_with("Sepal"))

# starts_with(): Starts with a prefix.

# ends_with(): Ends with a suffix.

# contains(): Contains a literal string.

# matches(): Matches a regular expression.

# num_range(): Matches a numerical range like 1:100.

# rename 選取所有行/更改欄位名稱 -----
rename(flights, ActualDepatureTime = dep_time) # rename 選取所有資料行

# distinct 選取不重覆資料 -----
set.seed(168)

df <- data.frame(
  x = sample(10, 100, replace = TRUE), # rep = replace
  y = sample(10, 100, rep = TRUE)
) # 100*2

head(df)

distinct(df)

nrow(distinct(df))

nrow(distinct(df, x, y))

distinct(df, x) # 與下列結果相同 unique(df$x)

distinct(df, y) # 與下列結果相同 unique(df$y)

# mutate 新增欄位,保留原資料 -----
mutate(mtcars, displ_l = disp / 61.0237)

# Chapter 13.正規表示式 -----
?regex
# https://jfjelstul.github.io/regular-expressions-tutorial/
# stringr 套件: https://stringr.tidyverse.org/

# Chapter 14.YouBike2.0臺北市公共自行車即時資訊資料分析 -----

# 資料來源: https://data.gov.tw/dataset/137993

# 載入套件
library(jsonlite)
library(dplyr)
library(leaflet)
library(htmltools)

# 匯入資料並轉換為tibble
urls <- "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
df <- tibble::as_tibble(fromJSON(txt = urls))

# 轉換為 factor
df$sarea <- factor(df$sarea)

# 轉換為 Date
df$infoDate <- as.Date(df$infoDate)

# 將 "YouBike2.0_" 取代為空白
df$sna <- gsub("YouBike2.0_", "", df$sna)

# 新增使用率變數 usage
df$usage <- (df$total - df$available_rent_bikes)/df$total

# 資料結構
str(df)

# 資料結構
str(df)

# 資料摘要
summary(df)

# 可借數量直方圖
tmp <- hist(df$available_rent_bikes)
ymax <- tmp$counts[1]+150
hist(df$available_rent_bikes, 
     ylim =c(0, ymax),
     xlab = "Available Rent Bikes",
     main = paste0("臺北市Youbike可借數量直方圖, ", df$infoDate[1]))
box()
grid()

# 可借數量依行政區水平長條圖
df_available_tmp <- aggregate(x = available_rent_bikes ~ sarea,
                              data = df,
                              FUN = sum)

df_available_tmp <- df_available_tmp[order(df_available_tmp$available_rent_bikes),]

df_available <- as.table(df_available_tmp$available_rent_bikes)
names(df_available) <- df_available_tmp$sarea
df_available

barplot(df_available, 
        width = 0.5, 
        horiz = TRUE, 
        cex.names = 0.8,
        las = 2,
        main = paste0("臺北市Youbike可借數量依行政區水平長條圖, ", df$infoDate[1]))
box()
grid()

# 臺北市Youbike使用率互動式地圖

# 設定地圖的標題
tag.map.title <- tags$style(HTML("
  .leaflet-control.map-title { 
    transform: translate(-50%,20%);
    position: fixed !important;
    left: 50%;
    text-align: center;
    padding-left: 10px; 
    padding-right: 10px; 
    background: rgba(255,255,255,0.75);
    font-weight: bold;
    font-size: 18px;
  }
"))

# 建立 tag 物件
title <- tags$div(
  tag.map.title, HTML("臺北市Youbike使用率互動式地圖")
)  

# leaflet地圖
m <- df %>% 
  leaflet() %>%
  addTiles() %>%
  addCircles(lng = ~longitude, 
             lat = ~latitude, 
             radius = ~usage*80, 
             label = ~paste0(df$sarea, "-", df$sna, "-", round(df$usage*100,0), "%"),
             labelOptions = labelOptions(
                   textsize = "16px",
                   style = list("font-weight" = "bold", padding = "4px"))) %>%
  addControl(title, position = "topleft", className="map-title")
m

# Chapter 15.RMarkdown輸出中文PDF錯誤 -----

# YouTube: https://youtu.be/6Wc75BH02iE
# LINK: https://rwepa.blogspot.com/2024/09/rmarkdown-chinese-font-pdf.html

# 解決方法:
# 步驟1. 安裝 tinytex
install.packages("tinytex")

# 步驟2. 安裝 TinyTeX
tinytex::install_tinytex()

#步驟3. 設定 .Rmd 檔案的標題 header-includes:, 本例以中文輸入測試
---
title: "大數據分析"
author: "李明昌"
date: "2024-07-27"
header-includes:
  - \usepackage{ctex}
output:
  pdf_document: default
  html_document: default
  word_document: default
---
# 第1次執行時, 將自動安裝 platex 相關套件, 安裝完成後自動建立PDF檔案

# Mathematics in R Markdown 
# https://rpruim.github.io/s341/S19/from-class/MathinRmd.html

# Chapter 16.JAVA_HOME設定

步驟1. 下載 Java SE Development Kit 21.0.8
https://download.oracle.com/java/21/latest/jdk-21_windows-x64_bin.exe

步驟2. 安裝 JDK
jdk-21_windows-x64_bin.exe

步驟3. 設定 JAVA_HOME
設定 \ 系統 \ 進階系統設定 \ 環境變數 \ 系統變數
新增系統變數 \ 變數名稱: JAVA_HOME , 變數值: C:\Program Files\Java\jdk-21 \ 確定

步驟4. 設定 Path
系統變數 \ Path \ 編輯 \ %JAVA_HOME%\bin \ 確定

步驟5. 命令提示字元
java -version
# java version "21.0.8" 2025-07-15 LTS ...
# end
