library(xgboost)
ready = c(3,6,11)
#ready = c(1:10)

#models = list()
for (i in ready) {
  models[[i]] = readRDS(paste("mx",i,".rda",sep = ""))
}

##############
# Prediction #
##############
load("test_processed.rda")
#load("test.id.rda")
test.data.matrix = data.matrix(test.p)
gc()
#targets.test <- data.frame(ID=test.p$VAR_0002)
#targets.test$VAR_0002 <- NULL

for (rows in split(1:145232, ceiling((1:145232)/40000))) {
  gc()
  for (i in ready) {
    gc()
    targets.test[rows, paste("x",i,sep = "")] <- predict(models[[i]], test.data.matrix[rows,],missing=NA)
  }
}
rm(test.data.matrix)

################
# Prediction m #
################
#mm6 = readRDS("../forestMail/m6.rda")
load("test_processed.rda")
#load("test.id.rda")
test.data.matrix = data.matrix(test.p)
gc()
#targets.test <- data.frame(ID=test.p$VAR_0002)
#targets.test$ID <- NULL
#targets.test.esm = data.frame(targets.test[,1])
#targets.test.esm$targets.test...1.=NULL

for (rows in split(1:145232, ceiling((1:145232)/40000))) {
  gc()
  targets.test[rows,"m6"] <- predict(models[[13]], test.data.matrix[rows,],missing=NA)
}
rm(test.data.matrix)

########################
# Submision Individual #
########################
for (i in ready){
  gc()
  output = data.frame("ID" = test.id, "target" = targets.test[paste("x",i,sep = "")])
  write.csv(output,paste("data/xgb4x",i,".csv",sep = ""),row.names=FALSE,quote=FALSE)
}

#################################
# Submision Individual m models #
#################################

output = data.frame("ID" = test.id, "target" = targets.test["m6"])
write.csv(output,paste("data/xgbm6.csv",sep = ""),row.names=FALSE,quote=FALSE)

######################
# Submision ensemble #
######################
attach(targets.test)

targets.test.esm["ex1"] = combinator(targets.test,
                                       c("x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"),
                                       c(1,1,1,1,1,1,1,1,1,1))
targets.test.esm["ex1"] = range01(targets.test.esm["ex1"])

targets.test.esm["ex1.1"] = combinator(targets.test,
                                       c("x1","x11","x3","x4","x5","x6","x7","x8","x9","x10"),
                                       c(1,1,1,1,1,1,1,1,1,1))
targets.test.esm["ex1.1"] = range01(targets.test.esm["ex1.1"])

targets.test.esm["e1.1"] = combinator(targets.test,
                                       c("x1","x11","x3","x4","x5","x6","x7","x8","x9","x10","m6"),
                                       c(1,1,1,1,1,1,1,1,1,1,1))
targets.test.esm["e1.1"] = range01(targets.test.esm["e1.1"])

targets.test.esm["e1.2"] = combinator(targets.test,
                                      c("x1","x11","x3","x4","x5","x6","x7","x8","x9","x10"),
                                      c(1,1,1,1,1,1,1,1,1,1))
targets.test.esm["e1.2"] = range01(targets.test.esm["e1.2"])

targets.test.esm["e1.3"] = combinator(targets.test,
                                      c("x1","x11","x3","x4","x5","x6","x7","x8","x9","x10"),
                                      c(1,1,1,1.1,1,.9,1,.9,1,1.1))
targets.test.esm["e1.3"] = range01(targets.test.esm["e1.3"])


targets.test.esm["glm.1"] = combinator(targets.test,
                                       c("x3","x7","x8","x9","x4","x6"),
                                       c(0.17238,0.03293,0.28925,0.15540,0.32750,0.49816))
targets.test.esm["glm.1"] = range01(targets.test.esm["glm.1"])

targets.test.esm["glm.2"] = combinator(targets.test,
                                       c("x3","x7","x8","x9","x4","x6","m6"),
                                       c(0.278596006,0.141634238,0.28925,0.15540,0.32750,0.49816,0.019725000))
targets.test.esm["glm.2"] = range01(targets.test.esm["glm.2"])

targets.test.esm["e2"] = combinator(targets.test,
                                    c("x1","x2","x11","x3","x4","x5","x6","x7","x8","x9","x10","m6"),
                                    c(1,.2,1,1,1.2,1,1,1,1.2,1.2,1.4))
targets.test.esm["e2"] = range01(targets.test.esm["e2"])

targets.test.esm["e2"] = combinator(targets.test,
                                    c("x1","x2","x11","x3","x4","x5","x6","x7","x8","x9","x10","m6"),
                                    c(1,1,1,1,1,1,1,1,1,1))
targets.test.esm["e2"] = range01(targets.test.esm["e2"])

#(Intercept)           x3           x7           x8           x9           x4           x6
#-0.10637      0.17238      0.03293      0.28925      0.15540      0.32750      0.49816

#targest.test["ex2"] = 10*x1 + 10*x2 + 10*x3 + 10*x4 + 10*x5 + 10*x6 + 10*x7 + 10*x8 + 10*x9 + 10*x10
#targest.test["ex2"] = range01(targest.test["ex2"])

targets.test["e1"] = (targets.test$x10+targets.test$m6+targets.test$x4)
targets.test["e1"] = range01(targets.test["e1"])

range01 <- function(x){(x-min(x))/(max(x)-min(x))}
######################
# Submision ensemble #
######################
name = "e1.3"
output = data.frame("ID" = test.id, "target" = targets.test.esm[name])
names(output) = c('ID',name)
write.csv(output,paste("data/xgbeX",name,".csv",sep = ""),row.names=FALSE,quote=FALSE)
