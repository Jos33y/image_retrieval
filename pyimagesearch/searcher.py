# import the necessary packages
import numpy as np
import csv


class Searcher:
    def __init__(self, index_path):
        # store our index path
        self.indexPath = index_path

    def search(self, query_features, limit=10):
        # initialize our dictionary of results
        results = {}

        # open the index file for reading

        with open(self.indexPath) as f:
            # initialize the CSV reader
            reader = csv.reader(f)

            # loop over the rows in the index
            for row in reader:
                # parse out the image ID and features, then compute the
                # chi-squared distance between the features in our index
                # and our query features
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, query_features)

                # now that we have the distance between the two feature
                # vectors, we can update the results dictionary -- the
                # key is the current image ID in the index and the
                # value is the distance we just computed, representing
                # how 'similar' the image in the index is to our query
                results[row[0]] = d

            # close the reader
            f.close()

        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])

        # return our (limited) results
        return results[:limit]

    def chi2_distance(self, hist_a, hist_b, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(hist_a, hist_b)])

        # return the chi-squared distance
        return d
