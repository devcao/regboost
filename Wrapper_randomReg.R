

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
randomReg.fit = function(x, y, colsample = 0.7, subsample = 1, holdvar = "default", n_reg = 500, 
                         lambda = 0.01, weight_metric = "rmse", intercept = TRUE, interaction = 2){
  
  if( !(is.numeric(interaction) && length(interaction) == 1) ) { stop("invalid interaction") }
  
  if (interaction > 1){
    
    formula = paste0("~0+", paste(paste0(".^", 2:interaction, collapse="+"))) %>% as.formula()
    x = model.matrix(formula, data = x %>% as.data.frame())
  
  }
  
  if(is.character(holdvar) && holdvar == "default" && interaction > 1){
    holdvar = 0:(ncol(x)-1)
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
  
  if (colsample > 1 || colsample <= 0){
    warning("invalid colsample: reset to default")
    colsample = 1
  }
  
  if (subsample > 1 || subsample <= 0){
    warning("invalid subsample: reset to default")
    subsample = 1
  }

  
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



#' Cross Validation engine for randomRegression model
#'
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param mtry int, number of features in use for each linear regression.
#' @param nfolds int, number of folds for cross validation, default 5.
#' @param n_threads int, number of threads for parallel computing # Under developing
#' @param ...
#' @return A vector contains RMSE for each cv folds.
#' @examples
#'
cv4_randomReg = function(x, y, nfolds = 10, n_threads = -1, ...){

  n = nrow(x)
  p = ncol(x)
  foldid = createFolds(1:n, k = nfolds)
  
  
  lapply(1:nfolds, function(fold){
    val_x = x[foldid[[fold]], ]
    val_y = y[foldid[[fold]]] 
    train_x = x[-foldid[[fold]], ]
    train_y = y[-foldid[[fold]]]
    
    rr_fit = randomReg.fit(x = train_x, y = train_y, holdvar = "default", ...)
    predict(rr_fit, newx = val_x, newy = val_y)$rmse

  }) %>% unlist %>% return()
  
}



#' CV Tuning mtry for randomRegression model
#'
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param mtry_grid vector, specify candidate values of mtry.
#' @param nfolds int, number of folds for cross validation, default 5.
#' @param n_threads int, number of threads for parallel computing # Under developing
#' @param plot_cv bool, if true, cross validation plot will be drawn.
#' @param ...
#' @return A list of cv results and best mtry.
#' @examples
#'
tune4_randomReg = function(x, y, nfolds = 10, tune_var, tune_grid, n_threads = -1, ...){
  
  cv_res = lapply(X = params_grid, FUN=cv4_randomReg, x = x, y = y, nfolds = nfolds, ...) 
  
  cv_mean = cv_res %>% lapply(mean) %>% unlist()
  cv_sd = cv_res %>% lapply(sd) %>% unlist()
  cvup = cv_mean + cv_sd
  cvlo = cv_mean - cv_sd
  cv_result = data.frame(cvm = cv_mean, cvsd = cv_sd, cvup = cvup, cvlo = cvlo)
  
  cv_min = min(cv_mean)
  cv_1se = cv_min + sd(cv_sd)
  mtry_min = mtry_grid[which.min(cv_mean)]
  
  
  cl <- match.call()
  cl[[1]] <- as.name("cv.randomRegression")
  out = list(call = cl,
             cv_result = cv_result, 
             mtry_min = mtry_min)
  class(out) = "cv.randomRegression"
  
  return(out)
  
  
}




#' CV Tuning mtry for randomRegression model
#'
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param mtry_grid vector, specify candidate values of mtry.
#' @param nfolds int, number of folds for cross validation, default 5.
#' @param n_threads int, number of threads for parallel computing # Under developing
#' @param plot_cv bool, if true, cross validation plot will be drawn.
#' @param ...
#' @return A list of cv results and best mtry.
#' @examples
#'
gridSearch_randomReg = function(x, y, nfolds = 10, params_list, n_threads = -1){
  # TODO:
  # Arguments:
  # Output:
  
  params_grid = cross(list(colsample = seq(0.1,1,0.1), ) )
  
  cv_res = lapply(X = params_grid, FUN=cv4_randomReg, x = x, y = y, nfolds = nfolds) 
  
  cv_mean = cv_res %>% lapply(mean) %>% unlist()
  cv_sd = cv_res %>% lapply(sd) %>% unlist()
  cvup = cv_mean + cv_sd
  cvlo = cv_mean - cv_sd
  cv_result = data.frame(cvm = cv_mean, cvsd = cv_sd, cvup = cvup, cvlo = cvlo)
  
  cv_min = min(cv_mean)
  cv_1se = cv_min + sd(cv_sd)
  mtry_min = mtry_grid[which.min(cv_mean)]



  return(list(cv_result = cv_result, mtry_min = mtry_min))
  
  
}

#### make a list of tuning parameters: apply(expand.gird(params_grid), 1, as.list)


plot.cv.randomRegression = function(object){
  
  if (!inherits(object, "cv.randomRegression"))
    stop("object not of class cv.randomRegression")
  
  cv_result = obj$cv_result
  if(plot_cv){
    p <- ggplot(cv_result, aes(x=mtry, y=cvm)) + 
      geom_point(size = 4, colour = "red")+
      geom_errorbar(aes(ymin=cvlo, ymax=cvup), width=.2, colour = "grey", 
                    position=position_dodge(0.05)) + 
      geom_vline(xintercept = mtry_min, linetype = "dashed", color = "grey") +
      theme(legend.position = "none")
    print(p)
  }
  
}


#' Gradient Boosting by
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param test_x matrix or data.frame specify the testing features.
#' @param n_rounds int, number of gradient boosting rounds.
#' @param ...
#' @return A vector of predictions for test features
#' @examples
#'
rfboost = function(x, y, n_rounds = 5, ntree = 500, ...){
  
  pred = 0
  
  for(i in 1:n_rounds){
    rf_fit = randomForest(x = x, y = y, ntree = ntree, ...)
    y = y - rf_fit$predicted
    pred = pred + predict(rf_fit, test_x)
  }
  
  return(pred)
  
  
}  



#' Gradient Boosting by combining randomRegression and randomForest
#'
#'
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param test_x matrix or data.frame specify the testing features.
#' @param n_rounds int, number of gradient boosting rounds.
#' @param ...
#' @return A vector of predictions for test features
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
             eta = eta)
  class(out) = "regboost"
  
  return(out)

} 

#' Gradient Boosting by combining randomRegression and randomForest
#'
#'
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param test_x matrix or data.frame specify the testing features.
#' @param n_rounds int, number of gradient boosting rounds.
#' @param ...
#' @return A vector of predictions for test features
#' @examples
#'
regboost.train = function(x, y, nfolds = 10, params = list(),
                          early_stopping_rounds = 2){
  
  if(length(params) == 0){params = list(n_rounds = 10, eta = 1, 
                                        rr_start = TRUE, 
                                        rr.control = list(), 
                                        rf.control = list())}
  n = nrow(x)
  p = ncol(x)
  foldid = createFolds(1:n, k = nfolds)
  lapply(1:nfolds, function(fold){
    val_x = x[foldid[[fold]], ]
    val_y = y[foldid[[fold]]] 
    train_x = x[-foldid[[fold]], ]
    train_y = y[-foldid[[fold]]]
    
    cv_rr_fit = do.call(regboost.train, args = c(list(x = x, y = y), rf.control) )
    predict(rr_fit, newx = val_x, newy = val_y)$rmse
    
    })
  
  x, y, n_rounds = 5, eta = 1, rr_start = TRUE, rr.control, rf.control, watchlist = list()
  
}
  


#' Gradient Boosting by combining randomRegression and randomForest
#'
#'
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param test_x matrix or data.frame specify the testing features.
#' @param n_rounds int, number of gradient boosting rounds.
#' @param ...
#' @return A vector of predictions for test features
#' @examples
#'
predict.regboost = function(object, newx, newy){
  
  if (!inherits(object, "regboost"))
    stop("object not of class regboost")
  
  eta = object$eta
  booster_info = object$booster_info


  pred = predict(booster_info[[1]], newx)$pred
  #pred = 0
  
  for(i in 2:object$n_rounds){
    pred = pred + predict(booster_info[[i]], newx) * eta
    cat(metric_fun(newy, pred, metric = "rmse"), "\n")
  }
  
  if (!missing(newy)) {
    newy = as.matrix(newy)
    rmse = metric_fun(newy, pred, metric = "rmse")
  }else{
    rmse = NULL
  }
  
  
  return(list(pred = pred, rmse = rmse))
  
}  




#' Gradient Boosting using randomForest + randomRegression
#'
#' @param x matrix or data.frame contains all training features.
#' @param y matrix or data.frame specify the training labels.
#' @param test_x matrix or data.frame specify the testing features.
#' @param reg_mtry: int, mtry for randomRegression.
#' @param n_rounds int, number of gradient boosting rounds.
#' @param ...
#' @return A vector of predictions for test features
#' @examples
#'
random_boosting = function(x, y, test_x, reg_try, holdvar, n_reg, n_round, ...){
  # TODO:
  # Argument:
  # Output:
  rr_fit = randomReg.fit(x = x, y = y, mtry = reg_try, n_reg = n_reg, holdvar = holdvar, ...)
  res = y - predict(rr_fit, x)$pred
  pred = predict(rr_fit, test_x)$pred
  
  rfb_pred = rfboost(x = x, y = res, test_x, n_rounds = n_round)
  
  pred = pred + rfb_pred
  return(pred)
}





