###### Setup ######
ii=2
set.seed(999)
library(xgboost)
###################

##load Matrix##
xgb.data = xgb.DMatrix(paste("xgb.data.",ii,".data",sep = ""),missing=NA)
gc()
xgb.validation = xgb.DMatrix(paste("xgb.validation.",ii,".data",sep = ""),missing=NA)
gc()
watchlist = list(validation=xgb.validation, train=xgb.data)
gc()

##Set Parameter##
param <- list(  objective           = "binary:logistic",
                eta                 = 0.005,
                max_depth           = 22,
                subsample           = .9,
                colsample_bytree    = .9,
                min_child_weight    = 24,
                max_delta_step      = 4,
                gamma               = 4,
                base_score          = .2325468,
                eval_metric         = "auc",
                misssing            = NA,
                lambda              = 1,
                alpha               = 0
)

gc()
##Run training##
clfx <- xgb.train(   params              = param,
                     data                = xgb.data,
                     nrounds             = 5000,
                     verbose             = 2,
                     watchlist           = watchlist,
                     early.stop.round    = 100,
                     maximize            = T,
                     missing             = NA)

##Save Result##
save(clfx,file = paste("mx",ii,".x.rda",sep = ""))
saveRDS(clfx,file = paste("mx",ii,".999.rda",sep = ""))


