library(data.table)
library(bit64)
library(xgboost)
library(caret)
library(dplyr) # for %>%
library(Epi) #for AUC evaluator
library(lattice) #for cov matrix plot
library(Ckmeans.1d.dp) #for 1d clustering

set.seed(121)
##########
# Import #
##########
train=fread('data/train.csv', header = T, sep = ',')
test=fread('data/test.csv', header = T, sep = ',')


train = as.data.frame(unclass(train), stringsAsFactors=T) #target is last column
test  = as.data.frame(unclass(test), stringsAsFactors=T)

test["target"] = NA

total = rbind(train,test)
attach(total)
rm(train,test)

tt = dim(train)[1]
###########################
# Remove constant columns #
###########################
col_unique_size = sapply(total, function(x) length(unique(x)))

# sum(col_ct==1) # Number of constant columns: 5
total = total[col_unique_size!=1]
rm(col_unique_size)

################################################################
# Split the data into [id, target, numeric, categorical, date] #
################################################################

#######################################
# Result:
# =====================================
# Dataframes      Description
# =====================================
# train.num       Numeric features
# train.date      Date features
# train.factor    Categorical features
# train.target    Target values
# train.id        ID values
#######################################

total.factor = total[sapply(total, is.factor)]

#target
target = total$target
#id
id = total$ID
#factor and date
total.factor.char = data.frame(sapply(total.factor, as.character), stringsAsFactors = F)
total.date.char = total.factor.char[grep("JAN1|FEB1|MAR1", total.factor.char)]

total.date = sapply(total.date.char, function(x) strptime(x, "%d%B%y:%H:%M:%S", tz = "GMT"))
total.date = cbind.data.frame(total.date)
total.factor = total.factor[!colnames(total.factor) %in% colnames(total.date)]

#numeric
total.num = total[sapply(total, is.numeric)]
total.num["ID"]=NULL
total.num["target"]=NULL

#cleanup
rm(total.date.char,total.factor.char,total)

detach(total)

#############
# Date Data #
#############
# Create cyclical & displacement data

# Extract month and year as int from each feature
date.year  = cbind.data.frame(sapply(total.date, year))

date.wday   = cbind.data.frame(sapply(total.date, wday))
date.mday   = cbind.data.frame(sapply(total.date, mday))
date.yday  = cbind.data.frame(sapply(total.date, yday))

date.size  = date.year*365 + date.yday #size wrt days
date.size  = date.size - min(date.size[!is.na(date.size)]) + 1 #recalibrate

##
total.date = data.frame(date.year, date.yday, date.wday, date.size)
##

rm(date.size,date.yday,date.mday,date.wday,date.year)

##################
# Categoric Data #
##################
# For categorical data, missing data doe not need to be marked if one hot encoding is used as there will be no difference.

# Categorical features with less than 20 classes
factor.small = total.factor[sapply(sapply(total.factor,unique),length)<=20]
factor.big   = total.factor[!names(total.factor) %in% names(total.factor.small)]

# Selection based on binomial p-value. featrures with all class's p-value>0.2 are not considered.
#train.factor.small_selected = subset(train.factor.small,
#                                     select = -c(VAR_0008, VAR_0009, VAR_0010, VAR_0011,
#                                                 VAR_0012, VAR_0043, VAR_0044, VAR_0196,
#                                                 VAR_0202, VAR_0214, VAR_0216, VAR_0222,
#                                                 VAR_0229, VAR_0230))
#test.factor.small = test.factor[names(test.factor) %in% names(train.factor.small)]
#test.factor.big = test.factor[names(test.factor) %in% names(train.factor.big)]
#test.factor.small_selected = test.factor[names(test.factor) %in% names(train.factor.small_selected)]


# bin.test_value = sapply( train_char.mini, function(x) sort(binom.test.multi(train_target_df,1,x), decreasing = T))
# full 1
# 8,9,10,11,12,43,44,196,202,216,222,229,230
# bin.test_value.impt = sapply(train_char.mini.impt, function(x) sort(binom.test.multi(train_target_df,1,x), decreasing = T))


# Manual One-Hot Encoder for mall unique
factor.small.uniques = apply(factor.small, unique, MARGIN = 2)

factor.small.decoded = data.frame(target)
factor.small.decoded['target'] = NULL

#For each feature
for (col.name in names(factor.small)) {
  #For each class in feature
  for (class.name in factor.small.uniques[[col.name]]) {
    factor.small.decoded[paste(col.name,class.name,sep = "_")] = as.numeric(factor.small[,col.name] == class.name)
  }
}

rm(factor.small.uniques)

## big factors
attach(factor.big)
# Manual remove
factor.big["VAR_0214"] = NULL #low info
factor.big["VAR_0404"] = NULL #high classes
factor.big["VAR_0493"] = NULL #high classes

#factor.big.char = data.frame(sapply(factor.big, as.character), stringsAsFactors = F)

factor.big.decoded = data.frame(target)
factor.big.decoded['target'] = NULL

# encode wrt probability# encode wrt probability
#transformation with low pass filter(remove low counts to reduce variance)
for (col.name in names(factor.big)) {
  factor.big.prop = prop.table(table(target,factor.big[,col.name],useNA = c("ifany")),margin = 2)[2,]
  for (factor.name in names(factor.big.prop)) {
    if (sum(factor.big[col.name]==factor.name)>40) {
      factor.big.decoded[factor.big[col.name]==factor.name,paste(col.name,"prop",sep = "_")] = factor.big.prop[factor.name]
    }
  }
}

#evaluation
sapply(factor.big.decoded, function(x) cor(target,x,use = "complete.obs"))

total.factor = data.frame(factor.small.decoded,factor.big.decoded)

#cleanup
rm(factor.big,factor.small,factor.big.prop,factor.small.uniques,factor.big.decoded,factor.small.decoded)

################
# Numeric Data #
################

#WIP
#
attach(total.num)
sort(sapply(total.num, function(x) sum(x==0,na.rm = T)))
sort(sapply(total.num, function(x) sum(x==-1,na.rm = T)))
sort(sapply(total.num, function(x) sum(x==-2,na.rm = T)))
sapply(total.num, function(x) sum(is.na(x)))
sort(sapply(train.num, function(x) sum(is.na(x))))

sort(sapply(train.num, function(x) sum( x == 999999998 )))

################
# Combine data #
################

train.data.full = data.frame(train.factor.small_selected.decoded,
                             train.num,
                             train.date.year_and_month)

test.data.full = data.frame(test.factor.small_selected.decoded,
                            test.num,
                            test.date.year_and_month)

train.data.full.matrix = data.matrix(train.data.full)

test.data.full.matrix = data.matrix(test.data.full)

total.p = data.frame(total.num,total.factor,total.date)

###############
# Outlier fix #
###############

add_clusters =function(df,x,interval=c(3,30)) {
  #get cor mat
  cormat = cor.mat(df,x)
  # set NA as 0
  cormat[is.na(cormat)] = 0
  cormat.sum = apply(cormat, 2, sum)
  #run ckmeans on sum
  ck.res=Ckmeans.1d.dp(cormat.sum,k=interval)
  print(ck.res)
  dfnew = data.frame(target)
  dfnew["target"] = NULL
  gc()
  #find intersection within cluster
  for (i in unique(ck.res$cluster) ) {
    cols = df[names(cormat.sum[ck.res$cluster==i])]
    col.size = dim(cols)[2]
    #create col from intersection
    dfnew[paste("outlier",x,i,sep="_")] = apply(cols,1,function(y) sum(y==x,na.rm = T))
    #filter if low info
    gc()
  }
  dfnew[dfnew==0]=NA
  dfnew
}

total.p.outliers = data.frame(target)
total.p.outliers["target"] = NULL
outliers = c(-1,999999999,999999998,999999997,999999996,999999995,999999994)

for (x in outliers) {
  cols = add_clusters(total.p,x,interval = c(2,40))
  total.p.outliers = cbind(total.p.outliers,cols)
  gc()
}

for (x in outliers) {
  total.p[total.p==x]=NA
  gc()
}

total.p=cbind(total.p,total.p.outliers)
################################
################################
cor.mat = function(df, x) {
  containsX = vector(length=dim(df)[2])
  for (i in seq(1,dim(df)[2])) {
    containsX[i] = x %in% df[,i]
  }
  gc()
  df.X = df[containsX]
  return(cor(df.X==x,use="complete.obs"))
}

cormap = function(df, x, sorted=T,clusters = c(5,30)) {
  containsX = vector(length=dim(df)[2])
  for (i in seq(1,dim(df)[2])) {
    containsX[i] = x %in% df[,i]
  }
  gc()
  df.X = df[containsX]
  cor.mat = cor(df.X==x,use="complete.obs")
  cor.mat[is.na(cor.mat)]=0
  cor.sum = apply(cor.mat,2, sum)

  df.X = df.X[names(sort(cor.sum))]
  cor.mat = cor(df.X==x,use="complete.obs")
  cor.mat[is.na(cor.mat)]=0

  print(Ckmeans.1d.dp(sort(cor.sum),k = clusters))
  return(cor.mat)

}

total.matrix = data.matrix(total.p)

containsX999=vector(length=145232)
containsX998=vector(length=145232)
i=0
for (col in total.p) {
  i=i+1
  print(i)
  containsX998[i] = 999999998 %in% col

}


total.p.999999999 = total.p[contains999999999]
total.p.X998 = total.p[containsX998]
total.p.X998 = total.p.X998==999999998

total.p.X998.s = total.p.999999998[names(sort(cm8.sum))]

cm =cor(total.p.999999999==999999999,use="complete.obs")
cm8=cor(total.p.X998,use="complete.obs")
cm8.sum = apply(cm8,2,sum)
cm8.sum2 = apply(cm8,2,function(x) sum(x[x>.1 | x<.1]))
cm8.s=cor(total.p.X998,use="complete.obs")


##############
# Manual fix #
##############

#sum(total.p.999999999[1:3]==999999999,na.rm = T)
#[1] 4 VAR0289



#####################
# Feature Reduction #
#####################
save(total.p,file="total_processed.rda")

#(Low pass filter) remove features that contains a class with porportion > .99 (magic)
lowPass = 0.999
total.size = dim(total.p)[1]
total.crit = total.size*lowPass
total.lowPassed95 = sapply(total,function(x){  max(table(x))<total.crit })
total.lowPassed999 = sapply(total.p,function(x){  max(table(x,useNA = c("ifany")))<total.size*0.999 })

total.lowPassed9999 = sapply(total.p,function(x){  max(table(x,useNA = c("ifany")))<total.size*0.9999 })

total.p = total.p[total.lowPassed9999]

train.p = total.p[!is.na(target),]
test.p  = total.p[is.na(target),]
save(test.p,file="test.p.rda")

rm(total.p)

####################
# train test split #
####################
load("total_processed.rda")

train.test.size.ess = floor(.10*145231)
train.index.ess = sample(seq(1:nrow(train.p)),size = train.test.size)

train.validation.ess  = train.p[train.index,]
target.validation.ess = target[!is.na(target)][train.index]

train.test.size = floor(.07*145231)
train.index = sample(seq(1:nrow(train.p)),size = train.test.size)

train.validation  = train.p[train.index,]
target.validation = target[!is.na(target)][train.index]

train.train  = train.p[-train.index,]
target.train = target[!is.na(target)][-train.index]

rm(total.p)

save(train.validation,file="train_validation1.rda")
save(target.validation,file="target_validation1.rda")
save(train.index,file="train_index1.rda")
###################
# XGB training T3 #
###################
models=list()

xgb.data       = xgb.DMatrix(data.matrix(train.train), label=target.train, missing = NA)
xgb.validation = xgb.DMatrix(data.matrix(train.validation), label=target.validation, missing = NA)
gc()
watchlist <- list(validation = xgb.validation, train = xgb.data)

param <- list(  objective           = "binary:logistic",
                eta                 = 0.005,
                max_depth           = 20,
                subsample           = .9,
                colsample_bytree    = .9,
                min_child_weight    = 20,
                max_delta_step      = 4,
                gamma               = 4,
                base_score          = .2325468,
                eval_metric         = "auc",
                misssing            = NA,
                lambda              = 1,
                alpha               = 0
)
# Removed:For warm start with saving of intermediate models.

for (i in 6:6) {
  clfx <- xgb.train(   params              = param,
                      data                = xgb.data,
                      nrounds             = 500000, #280, #125, #250, # changed from 300
                      verbose             = 2,
                      watchlist           = watchlist,
                      early.stop.round    = 75,
                      maximize            = T,
                      missing             = NA)
  models[[i]] = clfx
  #ptrain = predict(clf,xgb.data, outputmargin=T)
  #setinfo(xgb.data, "base_margin", ptrain)
}

param <- list(  objective           = "binary:logistic",
                eta                 = 0.01,
                max_depth           = 0,
                subsample           = 0.3,
                colsample_bytree    = 0.025,
                min_child_weight    = 10,
                max_delta_step      = 6,
                gamma               = 10,
                base_score          = .2325468,
                eval_metric         = "auc",
                misssing            = NA,
                lambda              = 1,
                num_parallel_tree   = 16
)
# Removed:For warm start with saving of intermediate models.

for (i in 6:6) {
  clfx <- xgb.train(   params              = param,
                       data                = xgb.data,
                       nrounds             = 500000, #280, #125, #250, # changed from 300
                       verbose             = 2,
                       watchlist           = watchlist,
                       early.stop.round    = 50,
                       maximize            = T,
                       missing             = NA)
  models[[i]] = clfx
  #ptrain = predict(clf,xgb.data, outputmargin=T)
  #setinfo(xgb.data, "base_margin", ptrain)
}
m6=clfx
gc()
param <- list(  objective           = "binary:logistic",
                eta                 = 0.01,
                max_depth           = 20,
                subsample           = 0.3,
                colsample_bytree    = 0.025,
                min_child_weight    = 10,
                max_delta_step      = 6,
                gamma               = 8,
                base_score          = .2325468,
                eval_metric         = "auc",
                misssing            = NA,
                lambda              = 1,
                num_parallel_tree   = 16
)
# Removed:For warm start with saving of intermediate models.

for (i in 7:7) {
  clfx <- xgb.train(   params              = param,
                       data                = xgb.data,
                       nrounds             = 500000, #280, #125, #250, # changed from 300
                       verbose             = 2,
                       watchlist           = watchlist,
                       early.stop.round    = 50,
                       maximize            = T,
                       missing             = NA)
  models[[i]] = clfx
  #ptrain = predict(clf,xgb.data, outputmargin=T)
  #setinfo(xgb.data, "base_margin", ptrain)
}


param <- list(  objective           = "binary:logistic",
                eta                 = 0.5,
                max_depth           = 20,
                subsample           = 0.9,
                colsample_bytree    = 0.9,
                min_child_weight    = 25,
                max_delta_step      = 4,
                gamma               = 4,
                base_score          = .2325468,
                eval_metric         = "auc",
                misssing            = NA
)
# Removed:For warm start with saving of intermediate models.

for (i in 6:6) {
  clfx <- xgb.train(   params              = param,
                       data                = xgb.data,
                       nrounds             = 3000, #280, #125, #250, # changed from 300
                       verbose             = 2,
                       watchlist           = watchlist,
                       early.stop.round    = 30,
                       maximize            = T,
                       missing             = NA)
  models[[i]] = clfx
  #ptrain = predict(clf,xgb.data, outputmargin=T)
  #setinfo(xgb.data, "base_margin", ptrain)
}


for (i in 5:16) {
  clf <- xgb.train(   params              = param,
                      data                = xgb.data,
                      nrounds             = 150, #280, #125, #250, # changed from 300
                      verbose             = 2,
                      watchlist           = watchlist,
                      early.stop.round    = 25,
                      maximize            = T,
                      missing             = NA
                      )
  models[[i]] = clf
  ptrain = predict(clf,xgb.data, outputmargin=T)
  setinfo(xgb.data, "base_margin", ptrain)
}
a=1
save(clf,file = "m1.rda")
load("m3.rda")
save(models,file="models.rda")
###################
# Test validation #
###################

prd.valid = predict(models[[3]],xgb.validation,missing=NA)
ROC(prd.valid,target.validation)

ROC(targets$e3x,target.validation.ess)

############
# Assemble #
############
xgb.validation2=xgb.validation
target.validation2=target.validation
load("target_validation1.rda")
load("train_validation1.rda")

xgb.validation.ess = xgb.DMatrix(data.matrix(train.validation.ess), label=target.validation, missing = NA)

xgb.validation.C12 = rbind(xgb.validation,xgb.validation2)

targets = data.frame(target.validation.ess)

targets["m1"] = predict(models[[1]],xgb.validation.ess,missing=NA)
targets["m3"] = predict(models[[3]],xgb.validation.ess,missing=NA)
targets["m6"] = predict(models[[6]],xgb.validation.ess,missing=NA)
attach(targets)
glm(target.validation.ess~m1+targets$m6,family=binomial())

targets["e1x"] = 3.526894*m1 + 2.138668*m3 -2.679326
targets["e2x"] = -2.6644768 + 3.6351896*m1+2.4411911*m3+0.7012364*m4-1.1547594*m5
targets["e3x"] = -5.395439 + 37.929297*m1+1.579263*m3-27.270880*targets$m6
targets[""]

##############
# Submission #
##############

load("test.p.rda")
test.data.full.matrix = data.matrix(test.p)

cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=test.p$VAR_0002)
submission$VAR_0002 <- NULL
# MAGICK
test_rows = 145232

for (rows in split(1:145232, ceiling((1:145232)/8000))) {
  gc()
  submission[rows, "target6"] <- predict(models[[6]], test.data.full.matrix[rows,],missing=NA)
}

cat("saving the submission file\n")

output = data.frame("ID" = test.id, "target" = submission$target6)
write.csv(output,"data/xgb4m6.csv",row.names=FALSE,quote=FALSE)

######################
# Submision ensemble #
######################

submission["e1"] = 3.526894*submission$target + 2.138668*submission$target3 -2.679326
submission["e1"] = range01(submission["e1"])


submission["e2"] = 3.6351896*submission$target + 2.4411911*submission$target3 + 0.7012364*submission$target4 -1.1547594*submission$target5 -2.6644768
submission["e2"] = range01(submission["e2"])


submission["e3"] = .15*submission$target + .10*submission$target3 + .75*submission$target6
submission["e3"] = range01(submission["e3"])

submission["e4"] = .25*submission$target + .10*submission$target3 + .65*submission$target6
submission["e4"] = range01(submission["e4"])

output = data.frame("ID" = id[is.na(target)], "target" = submission$e4)
write.csv(output,"data/xgb4e4.csv",row.names=FALSE,quote=FALSE)

################
# User Library #
################

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

table.fraction = function(column) {
  col_sum = apply(column, 2, sum)
  for (i in 1:dim(column)[1]) {
    column[i,] = column[i,]/col_sum
  }
  return(column)
}

# TODO: binomial p-test for features
# returns a list of p-values for each value in category
binom.test.multi = function(target,hit,feature) {
  p = sum(target==hit) / length(target)
  table.val = table(target,feature)
  return(apply(table.val,2,function(x) binom.test(x[2],x[1]+x[2],p,"t")$p.value))
}

quartiles = function(feature, bins) {
  as.integer(cut(feature, quantile(feature, probs=0:bins/bins,na.rm = T)))
}

table.prop = function(target,feature) {
  prop.table(table(target,feature,useNA = c("ifany")),margin = 2)
}

count.col_with = function(df,i) {
  sort(sapply(df, function(x) { sum(x==i) }))
}

swap