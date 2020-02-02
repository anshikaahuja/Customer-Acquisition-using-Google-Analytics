"""A simple example of how to access the Google Analytics API."""
# install google-api-python-client
# pip install google-auth-httplib2
# !pip install google-auth
# !pip install httplib2
# !pip install google

import argparse
import pandas as pd
import matplotlib. pyplot as plt
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error



from apiclient.discovery import build
import httplib2
from oauth2client import client
from oauth2client import file
from oauth2client import tools

class Service():
    def get_service(self, api_name, api_version, scope, client_secrets_path):
      """Get a service that communicates to a Google API.

      Args:
        api_name: string The name of the api to connect to.
        api_version: string The api version to connect to.
        scope: A list of strings representing the auth scopes to authorize for the
          connection.
        client_secrets_path: string A path to a valid client secrets file.

      Returns:
        A service that is connected to the specified API.
      """
      # Parse command-line arguments.
      parser = argparse.ArgumentParser(
          formatter_class=argparse.RawDescriptionHelpFormatter,
          parents=[tools.argparser])
      flags = parser.parse_args([])

      # Set up a Flow object to be used if we need to authenticate.
      flow = client.flow_from_clientsecrets(
          client_secrets_path, scope=scope,
          message=tools.message_if_missing(client_secrets_path))

      # Prepare credentials, and authorize HTTP object with them.
      # If the credentials don't exist or are invalid run through the native client
      # flow. The Storage object will ensure that if successful the good
      # credentials will get written back to a file.
      storage = file.Storage(api_name + '.dat')
      credentials = storage.get()
      if credentials is None or credentials.invalid:
        credentials = tools.run_flow(flow, storage, flags)
      http = credentials.authorize(http=httplib2.Http())

      # Build the service object.
      service = build(api_name, api_version, http=http)

      return service


    def get_first_profile_id(self, service):
      # Use the Analytics service object to get the first profile id.

      # Get a list of all Google Analytics accounts for the authorized user.
      print (service.management().accounts().list())
      accounts = service.management().accounts().list().execute()


      if accounts.get('items'):
        # Get the first Google Analytics account.
        account = accounts.get('items')[0].get('id')

        # Get a list of all the properties for the first account.
        properties = service.management().webproperties().list(
            accountId=account).execute()

        if properties.get('items'):
          # Get the first property id.
          property = properties.get('items')[0].get('id')

          # Get a list of all views (profiles) for the first property.
          profiles = service.management().profiles().list(
              accountId=account,
              webPropertyId=property).execute()

          if profiles.get('items'):
            # return the first view (profile) id.
            return profiles.get('items')[0].get('id')

      return None


    def get_results(self, service, profile_id,start_date,end_date,metrics,dimensions,view_id):
      # Use the Analytics Service Object to query the Core Reporting API
      return service.data().ga().get(
          ids='ga:' + view_id,
          start_date=start_date,
          end_date= end_date,
          metrics=metrics,
          dimensions=dimensions,
          max_results=100000).execute()

    def print_results(self, results):
      # Print data nicely for the user.
      if results:
        print ('View (Profile): %s' % results.get('profileInfo').get('profileName'))

      else:
        print ('No results found')

    def get_df(self, results,metrics,dimensions):
        # --getting results
        df = pd.DataFrame(results.get('rows'))

        # --getting col names
        n_dim = len(dimensions.split(','))
        n_met = len(metrics.split(','))
        col = dimensions.split(',')+metrics.split(',')
        col = [i[3:] for i in col] # --removing 'ga:' from col headers

        i=0
        for c in col:
            df.rename(columns={i:c},inplace=True)
            if c=='Date':
                df[c]=pd.to_datetime(df[c])
            elif i>(n_dim-1):
                df[c] = df[c].map(self.to_float)
            i=i+1
        return df

    def to_float(self,st):
        return float(st)

    def get_trend(self,data,x,metrics):
        n_met = len(metrics)
        fig, ax = plt.subplots(figsize=(15, 5))

        for met in metrics:
            plt.plot(data[x], data[met],label=met)
        plt.xlabel(x)
        plt.ylabel('metrics')
        plt.legend(loc='best')

    def get_resid_plot(self,actual,pred,norm=True):
        residual = actual-pred
        norm_resid = preprocessing.normalize([residual])
        resid_zero = np.zeros(len(residual))
        if norm:
            residual = norm_resid

        a,b = best_fit(pred,residual)
    #     print (len(a),len(b))
        yfit = [a + b * xi for xi in pred]

        fig, ax = plt.subplots(figsize=(15, 5))
    #     plt.plot(pred,resid_zero, color='red', linewidth=1, linestyle= '---')
        ax.scatter(pred, residual, c='black', alpha=0.3, edgecolors='none')
    #     plt.plot(pred,yfit, color='grey', linewidth=0.5, linestyle= '--')
        ax.legend()
        ax.grid(False)
        plt.xlabel('pred')
        plt.ylabel('residual')
        plt.legend(loc='best')

    def is_weekend(self,weekday):
        if weekday>4:
            return 1
        else:
            return 0

    # grid search sarima hyperparameters
    # one-step sarima forecast
    def sarima_forecast(self, history, config):
        order, sorder, trend = config
        # define model
        model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
        # fit model
        model_fit = model.fit(disp=False)
        # make one step forecast
        yhat = model_fit.predict(len(history), len(history))
        return yhat[0]

    # root mean squared error or rmse
    def measure_rmse(self,actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))

    # split a univariate dataset into train/test sets
    def train_test_split(self,data, n_test):
        return data[:-n_test], data[-n_test:]

    # walk-forward validation for univariate data
    def walk_forward_validation(self,data, n_test, cfg):
        predictions = list()
        # split dataset
        train, test = self.train_test_split(data, n_test)
    # 	print (test.shape, train.shape)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = self.sarima_forecast(history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
        # estimate prediction error
        error = self.measure_rmse(test, predictions)
        return error

    # score a model, return None on failure
    def score_model(self,data, n_test, cfg, debug=False):
        result = None
        # convert config to a key
        key = str(cfg)
        # show all warnings and fail on exception if debugging
        if debug:
            result = self.walk_forward_validation(data, n_test, cfg)
        else:
            # one failure during model validation suggests an unstable config
            try:
                # never show warnings when grid searching, too noisy
                with catch_warnings():
                    filterwarnings("ignore")
                    result = self.walk_forward_validation(data, n_test, cfg)
            except:
                error = None
        # check for an interesting result
        if result is not None:
            print(' > Model[%s] %.3f' % (key, result))
        return (key, result)

    # grid search configs
    def grid_search(self,data, cfg_list, n_test, parallel=True):
        scores = None
        if parallel:
            # execute configs in parallel
            executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
            tasks = (delayed(self.score_model)(data, n_test, cfg) for cfg in cfg_list)
            scores = executor(tasks)
        else:
            scores = [self.score_model(data, n_test, cfg) for cfg in cfg_list]
        # remove empty results
        scores = [r for r in scores if r[1] != None]
        # sort configs by error, asc
        scores.sort(key=lambda tup: tup[1])
        return scores

    # create a set of sarima configs to try
    def sarima_configs(self,p_params,d_params,q_params,t_params,P_params,D_params,Q_params,m_params):
        models = list()
        # create config instances
        for p in p_params:
            for d in d_params:
                for q in q_params:
                    for t in t_params:
                        for P in P_params:
                            for D in D_params:
                                for Q in Q_params:
                                    for m in m_params:
                                        cfg = [(p,d,q), (P,D,Q,m), t]
                                        models.append(cfg)
        return models
