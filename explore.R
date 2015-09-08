library(data.table)
library(bit64)
library(xgboost)
library(caret)

train=fread('data/train.csv', header = T, sep = ',')
test=fread('data/test.csv', header = T, sep = ',')


train = as.data.frame(unclass(train), stringsAsFactors=T) #target is last column
test = as.data.frame(unclass(test), stringsAsFactors=T)

tt = dim(train)[1]

###########################
# Remove constant columns #
###########################
col_ct = sapply(train, function(x) length(unique(x)))


# sum(col_ct==1) # Number of constant columns: 5
train = train[col_ct!=1]
test  = test[col_ct[1:length(col_ct)-1]!=1]

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

train.factor = train[sapply(train, is.factor)]
test.factor  = test[sapply(test, is.factor)]

train.target = train$target
target = train.target #alias

train.id = train$ID
test.id = test$ID

#train
train["ID"]=NULL
train["target"]=NULL

train.num = train[sapply(train, is.numeric)]
train.factor.char = data.frame(sapply(train.factor, as.character), stringsAsFactors = F)
train.date.char = train.factor.char[grep("JAN1|FEB1|MAR1", train.factor.char)]
train.factor = train.factor[!colnames(train.factor) %in% colnames(train.date.char)]

train.date = sapply(train.date.char, function(x) strptime(x, "%d%B%y:%H:%M:%S", tz = "GMT"))
train.date = cbind.data.frame(train.date)

#test
test["ID"]=NULL

test.num = test[sapply(test, is.numeric)]
test.factor.char = data.frame(sapply(test.factor, as.character), stringsAsFactors = F)
test.date.char = test.factor.char[grep("JAN1|FEB1|MAR1", test.factor.char)]
test.factor = test.factor[!colnames(test.factor) %in% colnames(test.date.char)]

test.date = sapply(test.date.char, function(x) strptime(x, "%d%B%y:%H:%M:%S", tz = "GMT"))
test.date = cbind.data.frame(test.date)

rm('test.factor.char','train.factor.char','test.date.char','train.date.char')


#############
# Date Data #
#############

# Extract month and year as int from each feature
train.date.year = cbind.data.frame(sapply(train.date, year))
train.date.month = cbind.data.frame(sapply(train.date, month))

test.date.year = cbind.data.frame(sapply(test.date, year))
test.date.month = cbind.data.frame(sapply(test.date, month))

test.date.year_and_month = data.frame(test.date.year, test.date.month)
train.date.year_and_month = data.frame(train.date.year, train.date.month)


# TODO: translate/recallibrate to 1

# Result:
# train.date.year_and_month    type:numeric dataframe
##################
# Categoric Data #
##################

# Categorical features with less than 20 classes
train.factor.small=train.factor[sapply(sapply(train.factor,unique),length)<=20]
train.factor.big= train.factor[!names(test.factor) %in% names(train.factor.small)]
# Selection based on binomial p-value. featrures with all class's p-value>0.2 are not considered.
train.factor.small_selected = subset(train.factor.small,
                                     select = -c(VAR_0008, VAR_0009, VAR_0010, VAR_0011,
                                                 VAR_0012, VAR_0043, VAR_0044, VAR_0196,
                                                 VAR_0202, VAR_0214, VAR_0216, VAR_0222,
                                                 VAR_0229, VAR_0230))

test.factor.small = test.factor[names(test.factor) %in% names(train.factor.small)]
test.factor.big = test.factor[names(test.factor) %in% names(train.factor.big)]
test.factor.small_selected = test.factor[names(test.factor) %in% names(train.factor.small_selected)]


# bin.test_value = sapply( train_char.mini, function(x) sort(binom.test.multi(train_target_df,1,x), decreasing = T))
# full 1
# 8,9,10,11,12,43,44,196,202,216,222,229,230
# bin.test_value.impt = sapply(train_char.mini.impt, function(x) sort(binom.test.multi(train_target_df,1,x), decreasing = T))


# Manual One-Hot Encoder
test.factor.small_selected.uniques = apply(test.factor.small_selected, unique, MARGIN = 2)
train.factor.small_selected.uniques = apply(train.factor.small_selected, unique, MARGIN = 2)

train.factor.small_selected.decoded = data.frame(target)
test.factor.small_selected.decoded = data.frame(test.id)

train.factor.small_selected.decoded['target'] = NULL
test.factor.small_selected.decoded['test.id'] = NULL

for (col.name in names(train.factor.small_selected)) {
  for (class.name in train.factor.small_selected.uniques[[col.name]]) {
    train.factor.small_selected.decoded[paste(col.name,class.name,sep = "_")] = as.numeric(train.factor.small_selected[,col.name] == class.name)
    test.factor.small_selected.decoded[paste(col.name,class.name,sep = "_")] = as.numeric(test.factor.small_selected[,col.name] == class.name)
  }
}

# Result:
# test.factor.small_selected.decoded    type:numeric dataframe

################
# Numeric Data #
################

#WIP
sum(sapply(train.num, function(x) x==0))

sort(sapply(train.num, function(x) sum(is.na(x))))

sort(sapply(train.num, function(x) sum( x == 999999998  )))

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

################
# XGB training #
################
# Adapted from https://www.kaggle.com/benhamner/springleaf-marketing-response/xgboost-example/code

clf =  xgboost(data        = train.data.full.matrix,
               label       = target,
               nrounds     = 100,
               objective   = "binary:logistic",
               verbose = 2,
               eval_metric = "auc",
               missing = NA
               )

###################
# XGB training T2 #
###################
#
xgb.data = xgb.DMatrix(train.data.full.matrix, label=target, missing = NA)


watchlist <- list(eval = xgb.data)
gc()
# Adapted from https://www.kaggle.com/michaelpawlus/springleaf-marketing-response/xgb-3/code
param <- list(  objective           = "binary:logistic",
                eta                 = 0.015,
                max_depth           = 8,
                subsample           = 0.9,
                colsample_bytree    = 0.9,
                min_child_weight    = 50,
                max_delta_step      = 3,
                gamma               = 3,
                base_score          = .23,
                eval_metric         = "auc",
                misssing            = NA
)

clf <- xgb.train(   params              = param,
                    data                = xgb.data,
                    nrounds             = 100, #280, #125, #250, # changed from 300
                    verbose             = 2,
                    watchlist           = watchlist,
                    early.stop.round    = 10,
                    missing             = NA)

###################
# XGB training T3 #
###################

# For warm start with saving of intermediate models.
models=list()
#19
for (i in 1:20) {
  moodels[[i]] = clf
  ptrain = predict(clf,xgb.data, outputmargin=T)
  setinfo(xgb.data, "base_margin", ptrain)

  clf <- xgb.train(   params              = param,
                      data                = xgb.data,
                      nrounds             = 100, #280, #125, #250, # changed from 300
                      verbose             = 2,
                      watchlist           = watchlist,
                      early.stop.round    = 10,
                      missing             = NA)
}

##############
# Submission #
##############

cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=test$ID)
submission$target <- NA

# MAGICK
test_rows = 145232

for (rows in split(1:145232, ceiling((1:145232)/8000))) {
  gc()
  submission[rows, "target"] <- predict(l[[2]], test.data.full.matrix[rows,],missing=NA)
}

cat("saving the submission file\n")

output = data.frame("ID" = submission$ID, "target" = submission$target)
write.csv(output,"data/xgb_nroundsX2_objBinLogistics_evalAuc.csv",row.names=FALSE,quote=FALSE)


################
# User Library #
################

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
