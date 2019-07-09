require(caret)
require(magrittr)
#' Fit randomRegression model
#'
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param mtry int, number of features in use for each linear regression.
#' @param holdvar ("default", "none") or a vector, specify features whihc will always be included in each linear regression, default -1. If holdvar = -1, no features will be hold.
#' @param n_reg int, total number of regressions, default 500.
#' @param lambda double, l2/ridge penalty factor for each regression,
#' @param weight_metric string, must be ("rmse", "mape", "none"), specify different methods for weighted ensembleing. "none" indicates no weights.
#' @param intercept bool, whether fit with intercept or not
#' @param interaction int, if less than 2, then no interaction term included. If > 2, specify the order of interactions. Default 2, all pairwise interactions included.
#' @return An randomRegression object.
#' @examples
#'
randomReg.fit = function(x, y, 
                         colsample = 0.7, subsample = 1, 
                         holdvar = "default", n_reg = 500, 
                         lambda = 0.01, weight_metric = "rmse", 
                         intercept = TRUE, interaction = 2, ...){
  if( !(is.numeric(interaction) && length(interaction) == 1) ) { stop("invalid interaction") }
  
  #cl = as.list(environment()); # cat("col: ", cl$colsample, "sub: ", cl$subsample, "n_reg: ", cl$n_reg, "\n");
  
  n_pred = ncol(x)
  if (interaction > 1){
    
    formula = paste0("~0+", paste(paste0(".^", 2:interaction, collapse="+"))) %>% as.formula()
    x = model.matrix(formula, data = x %>% as.data.frame())
  
  }
  
  if(is.character(holdvar) && holdvar == "default" && interaction > 1){
    holdvar = 0:(n_pred-1)
  }else if (is.character(holdvar) && holdvar == "default" && interaction <= 1){
    holdvar = -1
  }else if (is.character(holdvar) && holdvar == "none"){
    holdvar = -1
  }else if (is.character(holdvar)){
    stop("invalid holdvar, must be 'default', 'none' or a numeric vector")
  }else if (is.vector(holdvar)){
    if(any(is.na(holdvar))) {stop("NA not permitted in holdvar")}
    if(min(holdvar) <=0 || max(holdvar) > ncol(x)){
      stop("holdvar variable index out of valid range")
    }else{
      holdvar = holdvar - 1 
    }
  }else{
    stop("invalid holdvar")
  }
  
  x = as.matrix(x); y = as.matrix(y)
  
  
  n = nrow(x); p = ncol(x);
  if (length(y) != n) stop("length of response must be the same as predictors")
  
  lambda = as.double(lambda); n_reg=as.integer(n_reg); 
  if(missing(colsample)){ colsample = 0.7 }

  
  if(n_reg < 1){
    warning("invalid n_reg: reset to default: 500") 
    n_reg = 500L
  }
    
  
  if (any(is.na(x))) stop("NA not permitted in predictors")
  if (any(is.na(y))) stop("NA not permitted in response")
  

  ####
  rd_reg = randomRegression_fit(x = x, y = y, 
                       subsample = subsample,
                       colsample = colsample,
                       n_reg = n_reg, 
                       holdvar = holdvar,
                       lambda = lambda,
                       weight_metric = weight_metric,
                       intercept = intercept)
  
  ####
  cl <- match.call()
  cl[[1]] <- as.name("randomRegression")
  out = list(call = cl,
             rd_reg = rd_reg,
             interaction = interaction)
  
  class(out) = "randomRegression"
  return(out)
}



#' Predict for randomRegression model
#'
#' @param object A randomRegression object
#' @param newx matrix or data.frame contains all testing features.
#' @param newy matrix or data.frame specify the testing labels. Could be NULL.
#' @return A list of predictions and rmse. If newy is NULL, only predicted values returned.
#' @examples
#'
predict.randomRegression = function(object, newx, newy){
  
  if (!inherits(object, "randomRegression"))
    stop("object not of class randomRegression")
  
  if(is.vector(newx)){
    newx = matrix(newx, 1, length(newx))
  }else{
    newx = as.matrix(newx)
  }
  
  
  rd_obj = object[["rd_reg"]]
  
  if(object$interaction > 1) {
    formula = paste0("~0+", paste(paste0(".^", 2:object$interaction, collapse="+"))) %>% as.formula()
    newx = model.matrix(formula, data = newx %>% as.data.frame())
  }
  
  pred = randomRegression_predict(rd_obj, newx)
  
  if (!missing(newy)) {
    newy = as.matrix(newy)
    rmse = metric_fun(newy, pred, metric = "rmse")
  }else{
    rmse = NULL
  }
  
  return( list(pred = pred, rmse = rmse) )
  
}


#' Gradient Boosting by combining randomRegression and randomForest
#'
#'
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param test_x matrix or data.frame specify the testing features.
#' @param n_rounds int, number of gradient boosting rounds.
#' @param eta double, learning rate
#' @param rr_start bool, TRUE: use randomRegression as the initial, FALSE, randomForest initial
#' @param rr.control list of params for randomRegression
#' @param rf.control list of params for randomForest
#' @return A regboost object
#' @examples
#'
regboost.train = function(x, y, n_rounds = 5, eta = 1, rr_start = TRUE, rr.control, rf.control, watchlist = list()){
  
  booster_info = list()
  y_og = y
  
  
  if(rr_start){
    rr_fit = do.call(randomReg.fit, args = c(list(x = x, y = y), rr.control) )
    booster_info[[1]] = rr_fit
    rr_inSample = predict(rr_fit, x, y)#############
    
    y = y - rr_inSample$pred
    
    if(length(watchlist) >0){
      rr_pred = predict(rr_fit, newx=watchlist$xval, newy = watchlist$yval)
      val_err = rr_pred$rmse
    }else{
      val_err = NULL
    }
    cat("Now running round ", 1, "training error: ", rr_inSample$rmse, "validation err:", val_err, "\n")#########
    
    re_inSample_pred = rr_inSample$pred
    pred = rr_pred$pred
    
  }else{
    rr_fit = do.call(randomForest, args = c(list(x = x, y = y), rf.control) )
    booster_info[[1]] = rr_fit
    rr_inSample = predict(rr_fit, x)
    rr_inSample_rmse = metric_fun(y = y, y_hat = rr_inSample, metric = "rmse")
    y = y - rr_inSample
    
    if(length(watchlist) >0){
      rr_pred = predict(rr_fit, newdata=watchlist$xval, newy = watchlist$yval)
      val_err = metric_fun(y = watchlist$yval, y_hat = rr_pred, metric = "rmse")
    }else{
      val_err = NULL
    }
    cat("Now running round ", 1, "training error: ", rr_inSample_rmse, "validation err:", val_err, "\n")#########
    
    re_inSample_pred = rr_inSample
    pred = rr_pred
    
  }
    

  for(i in 2:n_rounds){
    rf_fit = do.call(randomForest, args = c(list(x = x, y = y), rf.control) )
    booster_info[[i]] = rf_fit
    y = y - rf_fit$predicted * eta
    
    re_inSample_pred = re_inSample_pred + rf_fit$predicted * eta
    
    if(length(watchlist) >0){
      pred = pred + predict(rf_fit, watchlist$xval) * eta
      val_err = metric_fun(y = watchlist$yval, y_hat = pred, metric = "rmse")
    }else{
      val_err = NULL
    }
    
    train_err = metric_fun(y = y_og, y_hat = re_inSample_pred, metric = "rmse")
    cat("Now running round ", i, "training error: ", train_err, "validation err:", val_err, "\n")
    
    
  }
  
  cl <- match.call()
  cl[[1]] <- as.name("regboost")
  out = list(call = cl,
             booster_info = booster_info,
             n_rounds = n_rounds, 
             eta = eta,
             rr_start = rr_start)
  class(out) = "regboost"
  
  return(out)

} 

#' Predict for regboost model
#'
#' @param object A regboost object
#' @param newx matrix or data.frame contains all testing features.
#' @param newy matrix or data.frame specify the testing labels. Could be NULL.
#' @return A list of predictions and rmse. If newy is NULL, only predicted values returned.
#' @examples
#'
predict.regboost = function(object, newx, newy){
  
  if (!inherits(object, "regboost"))
    stop("object not of class regboost")
  
  eta = object$eta
  booster_info = object$booster_info
  
  if(object$rr_start){
    pred = predict(booster_info[[1]], newx)$pred
  }else{
    pred = predict(booster_info[[1]], newx)
  }
  #pred = 0
  
  for(i in 2:object$n_rounds){
    pred = pred + predict(booster_info[[i]], newx) * eta
  }
  
  if (!missing(newy)) {
    newy = as.matrix(newy)
    rmse = metric_fun(newy, pred, metric = "rmse")
  }else{
    rmse = NULL
  }
  
  
  return(list(pred = pred, rmse = rmse))
  
}  



train_test_split = function(x, y, ratio){
  train_id = sample(1:nrow(x), ceiling(nrow(x)*ratio))
  x_train = x[train_id, ]; y_train = y[train_id,]
  x_test = x[-train_id, ]; y_test = y[-train_id,]
  
  return(list(x_train = x_train,
              y_train = y_train,
              x_test = x_test,
              y_test = y_test))
}


