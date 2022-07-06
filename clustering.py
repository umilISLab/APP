import typing
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def type_check(method):
    def onCall(*args, **kwargs):
        params_kwargs = kwargs

        # avoid self
        params_args = args[1:]
        param_types = typing.get_type_hints(method)
        keys = list(param_types.keys())
        params_kwargs.update(dict(zip(keys, params_args)))

        for k, v in params_kwargs.items():
            if k in ['self', 'return']:
                continue
            else:
                try:
                    if param_types[k].__class__.__module__ != 'typing' and \
                            not isinstance(v, param_types[k]) or \
                            (param_types[k].__class__.__module__ == 'typing' and \
                             not isinstance(v, param_types[k].__args__)):
                        raise Exception(f'{k} must be {param_types[k]} not {type(v)}')
                except:
                    print(param_types[k], k)

        return method(args[0], **params_kwargs)

    return onCall


class AffinityPropagation:

    @type_check
    def __init__(self,
                 affinity: object,
                 damping: float = 0.5,
                 max_iter: int = 200,
                 convergence_iter: int = 15,
                 copy: bool = True,
                 pc: float = 0.0015,
                 verbose: bool = False,
                 random_state: int = None,
                 aging_index: int = 0,
                 preference: float = None,
                 freezing: bool = False,
                 columns_to_scale: list = None,
                 columns_to_encode: list = None):
        """Check https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html for information on parameters."""

        # input parameters
        for k, v in locals().items():
            if k == 'self': continue
            setattr(self, k, v)

        ### Additional Attributes ###
        # time step index of data arriving
        self.time_step = 0

        # keep in memory data point
        self.X = None

        # keep in memory previous labels and clusters
        self.labels = dict()
        self.centroids = dict()
        self.cut = list()
        self.to_remove = list()

    def __step(self, ind: int, tmp: np.ndarray) -> tuple:
        '''
        This is meant to return the new Availability (A) and Repsonsibility (R) matrices for all the data points
        Args:
            tmp(object): array-like of shape (n_samples, n_samples), Intermediate results
            ind(int): tmp index
        Returns:
            R(object): array-like of shape (n_samples, n_samples). Responsibility matrix
            A(object): array-like of shape (n_samples, n_samples). Availability matrix
        '''
        N, N = tmp.shape  # == self.R.shape == self.N.shape
        old_R, old_A = self.R, self.A

        # R UPDATE STEP - from sklearn
        np.add(old_A, self.S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)
        # tmp = Rnew
        np.subtract(self.S, Y[:, None], tmp)
        tmp[ind, I] = self.S[ind, I] - Y2

        # Damping
        tmp *= 1 - self.damping
        R = old_R * self.damping
        R += tmp

        # A UPDATE STEP - from sklearn
        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[:: N + 1] = R.flat[:: N + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: N + 1] = dA

        # Damping
        tmp *= 1 - self.damping
        A = old_A * self.damping
        A -= tmp

        # TODO
        # if self.time_step>1 and self.freezing:
        #    A[:old_A.shape[0], :old_A.shape[0]] = old_A
        #    R[:old_R.shape[0], :old_R.shape[0]] = old_R

        return R, A

    @type_check
    def fit(self, X: np.ndarray) -> tuple:
        '''
        This runs the Incremental Affinity Propagation Clustering Based on Nearest Neighbor Assignment until convergence
        Args:
            X(object): array-like of shape (n_samples, n_samples). Training instances to cluster
        Returns:
            exemplar : ndarray. Cluster exemplars
            labels : ndarray of shape (n_samples,). Cluster labels
        '''
        self.number_of_iteration = 0

        # first data
        if self.time_step == 0:
            self.old_X = np.array([])
            self.X = X

            # Instantiate encoder/scaler
            scaler = MinMaxScaler()
            ohe = OneHotEncoder(sparse=False)

            # Scale and Encode Separate Columns
            if self.columns_to_scale is not None:
                scaled_columns = scaler.fit_transform(X.T[self.columns_to_scale].T)
            if self.columns_to_encode is not None:
                encoded_columns = ohe.fit_transform(X.T[self.columns_to_encode].T)

            # Concatenate (Column-Bind) Processed Columns Back Together
            if self.columns_to_encode is not None and self.columns_to_scale is not None:
                processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)
            elif self.columns_to_scale is not None:
                processed_data = scaled_columns
            elif self.columns_to_encode is not None:
                processed_data = encoded_columns
            else:
                processed_data = X

            self.X_norm = processed_data
            self.old_X_norm = np.array([])
            self.X_norm, self.X = self.X, self.X_norm

            # COMPUTE THE SIMILARITY MATRIX
            if self.affinity == 'precomputed':
                self.S = X
            else:
                self.S = self.affinity(X, X)

            # self.preference = self.get_preference()
            if self.preference is None:
                self.preference = np.median(self.S)

            N, N = self.S.shape

            # Place preference on the diagonal of S
            self.preference = np.array(self.preference)
            self.S.flat[:: (N + 1)] = self.preference

            # Remove degeneracies
            self.S = self.S.reshape(N, N)

            #  INIITALISE THE RESPONSIBILITY AND THE AVAILABILITY MATRICES
            N, N = self.S.shape
            self.R, self.A = np.zeros((N, N)), np.zeros((N, N))

            self.X_norm, self.X = self.X, self.X_norm

        # new data are available. Perform clustering from pre-computed matrices
        else:
            if self.freezing:
                min_ = self.S.min()
                max_ = self.S.max()
                self.S = np.array([[max_ if self.labels_[i] == self.labels_[j]
                                    else min_ for j in range(self.X.shape[0])] for i in range(self.X.shape[0])])
                # self.A = np.array([[1 if self.labels_[i]==self.labels_[j]
                #                    else self.A[i,j] for j in range(self.X.shape[0])] for i in range(self.X.shape[0])])
                # self.R = np.array([[0 if self.labels_[i]!=self.labels_[j]
                #                    else self.R[i,j] for j in range(self.X.shape[0])] for i in range(self.X.shape[0])])

            self.old_X = self.X

            # remove duplicates: data from timestep i that are already considered in timestep i-1
            if X.ndim == 1:
                new_X = np.setdiff1d(X, self.old_X)
            else:
                # TODO: remove duplicates
                new_X = X

            if len(new_X) == 0:
                warnings.warn("Incremental Affinity propagation needs new training instances.")
                return None, None

            self.X = np.concatenate([self.old_X, new_X])

            # Instantiate encoder/scaler
            scaler = MinMaxScaler()
            ohe = OneHotEncoder(sparse=False)

            # Scale and Encode Separate Columns
            if self.columns_to_scale is not None:
                scaled_columns = scaler.fit_transform(self.X.T[self.columns_to_scale].T)
            if self.columns_to_encode is not None:
                encoded_columns = ohe.fit_transform(self.X.T[self.columns_to_encode].T)

            # Concatenate (Column-Bind) Processed Columns Back Together
            if self.columns_to_encode is not None and self.columns_to_scale is not None:
                processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)
            elif self.columns_to_scale is not None:
                processed_data = scaled_columns
            elif self.columns_to_encode is not None:
                processed_data = encoded_columns
            else:
                processed_data = self.X

            self.X_norm = processed_data
            self.old_X_norm = self.X_norm[:self.old_X.shape[0]]
            new_X_norm = self.X_norm[self.old_X.shape[0]:]
            new_X, new_X_norm = new_X_norm, new_X
            self.X_norm, self.X = self.X, self.X_norm
            self.old_X, self.old_X_norm = self.old_X_norm, self.old_X

            # Similarity matrix S update
            S_tmp1 = self.affinity(new_X, self.old_X)
            S_tmp2 = self.affinity(new_X)
            self.S = np.concatenate([np.concatenate([self.S, S_tmp1.T], axis=1),  # up & right
                                     np.concatenate([S_tmp1, S_tmp2], axis=1)])  # down

            new_X, new_X_norm = new_X_norm, new_X
            self.X_norm, self.X = self.X, self.X_norm
            self.old_X, self.old_X_norm = self.old_X_norm, self.old_X

            # self.preference = self.get_preference()
            if self.preference is None:
                self.preference = np.median(self.S)

            N, N = self.S.shape

            # Place preference on the diagonal of S
            self.preference = np.array(self.preference)
            self.S.flat[:: (N + 1)] = self.preference

            # Remove degeneracies
            self.S = self.S.reshape(N, N)

            # Responsibility matrix R update
            old_R, old_A, old_N = self.R, self.A, self.R.shape[1]

            R_tmp1 = np.array([[self.__matrix_update(self.S, old_R, i, j) for j in range(old_N)]
                               for i in range(old_N, N)]).T
            R_tmp2 = np.array([[self.__matrix_update(self.S, old_R, i, j) for i in range(old_N)]
                               for j in range(old_N, N)])
            R_tmp3 = np.zeros((new_X.shape[0], new_X.shape[0]))
            self.R = np.concatenate([np.concatenate([old_R, R_tmp1], axis=1), np.concatenate([R_tmp2, R_tmp3], axis=1)])

            # Availability matrix A update
            A_tmp1 = np.array([[self.__matrix_update(self.S, old_A, i, j) for j in range(old_N)]
                               for i in range(old_N, N)]).T
            A_tmp2 = np.array([[self.__matrix_update(self.S, old_A, i, j) for i in range(old_N)]
                               for j in range(old_N, N)])
            A_tmp3 = np.zeros((new_X.shape[0], new_X.shape[0]))

            self.A = np.concatenate([np.concatenate([old_A, A_tmp1], axis=1), np.concatenate([A_tmp2, A_tmp3], axis=1)])

        # -- #
        # Check for convergence (from scikit-learn)
        e = np.zeros((N, self.convergence_iter))
        # -- #

        ind = np.arange(N)
        tmp = np.zeros((N, N))

        for i in range(self.max_iter):
            if self.verbose:
                print("processing iteration %d" % (i,))
            self.R, self.A = self.__step(ind, tmp)

            # -- #
            # Check for convergence (from scikit-learn)
            E = (np.diag(self.A) + np.diag(self.R)) > 0
            e[:, i % self.convergence_iter] = E
            K = np.sum(E, axis=0)

            if i >= self.convergence_iter:
                se = np.sum(e, axis=1)
                unconverged = np.sum((se == self.convergence_iter) + (se == 0)) != N
                if (not unconverged and (K > 0)) or (i == self.max_iter):
                    never_converged = False
                    if self.verbose:
                        print("Converged after %d iterations." % i)
                    self.n_iter_ = i
                    break
            # -- #

        else:
            never_converged = True
            self.n_iter_ = self.max_iter
            if self.verbose:
                print("Did not converge")

        I = np.flatnonzero(E)
        K = I.size  # Identify exemplars

        if K > 0 and not never_converged:
            c = np.argmax(self.S[:, I], axis=1)
            c[I] = np.arange(K)  # Identify clusters
            # Refine the final set of exemplars and clusters and return results
            for k in range(K):
                ii = np.where(c == k)[0]
                j = np.argmax(np.sum(self.S[ii[:, np.newaxis], ii], axis=0))
                I[k] = ii[j]

            c = np.argmax(self.S[:, I], axis=1)
            c[I] = np.arange(K)
            labels = I[c]
            # Reduce labels to a sorted, gapless, list
            cluster_centers_indices = np.unique(labels)
            labels = np.searchsorted(cluster_centers_indices, labels)
        else:
            warnings.warn(
                "Affinity propagation did not converge, this model "
                "will not have any cluster centers.")
            labels = np.array([-1] * N)
            cluster_centers_indices = []

        self.labels[self.time_step] = labels[self.old_X.shape[0]:]
        self.centroids[self.time_step] = cluster_centers_indices

        self.labels_ = labels
        self.cluster_centers_indices_ = cluster_centers_indices

        self.__forgot_meanings()

        self.time_step += 1

        return labels, cluster_centers_indices

    def __matrix_update(self, S, M, i, j):
        '''Matrix update for Incremental Affinity Propagation
        Args:
            S: similarity matrix
            M: matrix to update (Availability or Responsibility)
            i: row index
            j: column index
        '''

        # update as in the original paper
        old_N = M.shape[1]

        if i >= old_N:
            i = S[:old_N, i].argmax()
            return M[i, j]

        if j >= old_N:
            j = S[j, :old_N].argmax()
            return M[i, j]

    def __forgot_meanings(self):
        if self.time_step <= self.aging_index or self.aging_index == 0:
            self.cut.append(set())
            return

        # mapping
        index = 0
        for time_step in range(self.time_step):
            n = self.labels[time_step].shape[0]
            self.labels[time_step] = self.labels_[index: index + n]
            index += n

        cut = set()
        for time_step in range(self.time_step - 1 - self.aging_index, self.time_step - 1):
            for label in np.unique(self.labels[time_step]):
                outdate = True
                for time_step_next in range(time_step + 1, self.time_step):
                    if label in self.labels[time_step_next]:
                        outdate = False

                if outdate:
                    cut.add(label)
                    self.X = self.X[self.labels_ != label]
                    self.to_remove.append(np.where(self.labels_ == label)[0])
                    self.R = self.R[self.labels_ != label]
                    self.A = self.A[self.labels_ != label]
                    self.S = self.S[self.labels_ != label]
                    self.R = self.R[:, self.labels_ != label]
                    self.A = self.A[:, self.labels_ != label]
                    self.S = self.S[:, self.labels_ != label]
                    self.labels_ = self.labels_[self.labels_ != label]

                    for t in range(self.time_step - 1):
                        self.labels[t] = self.labels[t][self.labels[t] != label]
        self.cut.append(cut)


class PosterioriAffinityPropagation:
    def __init__(self, affinity: object, aging_index: float = 2, damping: float = 0.5, max_iter: int = 200,
                 convergence_iter: int = 15, preference: float = None, freezing: bool = True,
                 columns_to_scale: list = None,
                 columns_to_encode: list = None, exemplar_pack=True):
        self.time_tag = 0
        self.data = list()
        self.aging_index = aging_index
        self.damping = damping
        self.max_iter = max_iter
        self.affinity = affinity
        self.aging_index = aging_index
        self.damping = damping
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.freezing = freezing
        self.exemplar_pack = exemplar_pack # replace centroids with exemplar at the 0th step

        self.columns_to_scale = columns_to_scale
        self.columns_to_encode = columns_to_encode

        # keep in memory previous labels and clusters
        self.labels = dict()
        self.centroids = dict()
        self.cut = list()
        self.to_remove = list()

    def _pack(self, X: np.array, labels: np.array) -> (np.array, np.array):
        '''Pack clusters of vectors into single representations'''
        unique = np.unique(labels)
        return np.array([X[labels == label].mean(0) for label in unique]), unique

    def fit(self, X: np.array) -> None:

        self.data.append(X)

        # standard AP
        if self.time_tag == 0:
            ap = AffinityPropagation(damping=self.damping, max_iter=self.max_iter,
                                     convergence_iter=self.convergence_iter, affinity=self.affinity,
                                     preference=self.preference, columns_to_scale=self.columns_to_scale,
                                     columns_to_encode=self.columns_to_encode)
            ap.fit(X)
            self.n_iter_ = ap.n_iter_
            self.S = ap.S

            if not self.exemplar_pack:
                self._X_pack, self._labels_pack = self._pack(X, ap.labels_)
            else:
                self._X_pack, self._labels_pack = np.array([X[i] for i in ap.cluster_centers_indices_]), np.unique(ap.labels_)
            self.X_, self.labels_ = X, ap.labels_
            self.cluster_centers_indices_ = ap.cluster_centers_indices_

            # didnt converge or single cluster
            if np.unique(self.labels_).shape[0] <= 1:
                self.X_, self.labels_ = X, np.array([1] * X.shape[0])
                self._X_pack = np.array([X.mean(0)])
                self._labels_pack = np.array([1])

            self.labels[self.time_tag] = ap.labels_

        else:
            X_prev, X_curr = self.X_, X
            X_prev_pack = self._X_pack
            X_next = np.concatenate([X_curr, X_prev_pack])

            # preference over packed vectors
            preference = self.preference
            ap = AffinityPropagation(damping=self.damping, affinity='precomputed',
                                     max_iter=self.max_iter, preference=preference,
                                     convergence_iter=self.convergence_iter)
            if self.freezing:
                min_ = self.S.min()
                max_ = self.S.max()
                self.S = np.array([[min_ if i != j else max_ for i in range(X_prev_pack.shape[0])]
                                   for j in range(X_prev_pack.shape[0])])
            else:
                # Instantiate encoder/scaler
                scaler = MinMaxScaler()
                ohe = OneHotEncoder(sparse=False)

                # Scale and Encode Separate Columns
                if self.columns_to_scale is not None:
                    scaled_columns = scaler.fit_transform(X_next.T[self.columns_to_scale].T)
                if self.columns_to_encode is not None:
                    encoded_columns = ohe.fit_transform(X_next.T[self.columns_to_encode].T)

                # Concatenate (Column-Bind) Processed Columns Back Together
                if self.columns_to_encode is not None and self.columns_to_scale is not None:
                    processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)
                elif self.columns_to_scale is not None:
                    processed_data = scaled_columns
                elif self.columns_to_encode is not None:
                    processed_data = encoded_columns
                else:
                    processed_data = X_next

                self.X_norm = processed_data

                X_curr, X_prev_pack = self.X_norm[:X_curr.shape[0]], self.X_norm[X_curr.shape[0]:]

                if X_prev_pack.shape[0] == 1:
                    self.S = np.array([1]).reshape(-1, 1)
                else:
                    self.S = self.affinity(X_prev_pack)

            S_tmp1 = self.affinity(X_curr, X_prev_pack)
            S_tmp2 = self.affinity(X_curr, X_curr)
            self.S = np.concatenate([np.concatenate([self.S, S_tmp1.T], axis=1),  # up & right
                                     np.concatenate([S_tmp1, S_tmp2], axis=1)])

            ap.fit(self.S)

            X_curr, X_prev_pack, X_prev = X, self._X_pack, self.X_

            self.n_iter_ = ap.n_iter_

            self.cluster_centers_indices_ = ap.cluster_centers_indices_
            labels_prev, labels_next = self.labels_, ap.labels_
            labels_prev_pack = self._labels_pack
            new_labels_prev_pack = labels_next[:labels_prev_pack.shape[0]]
            labels_curr = labels_next[labels_prev_pack.shape[0]:]

            # map old pack labels to new label
            new_labels_prev = labels_prev.copy()
            for old_, new_ in zip(labels_prev_pack, new_labels_prev_pack):
                new_labels_prev[labels_prev == old_] = new_

            # trim t1|t0
            self._prev_X, self._prev_labels = self.X_, new_labels_prev
            self._prev_X_pack, self._prev_labels_pack = self._X_pack, new_labels_prev_pack
            X_next = np.concatenate([self._prev_X, X_curr])
            labels_next = np.concatenate([self._prev_labels, labels_curr])

            self._curr_X, self._curr_labels = X_next, labels_next
            self._curr_X_pack, self._curr_labels_pack = self._pack(X_next, labels_next)

            self.X_, self.labels_ = X_next, labels_next
            self._X_pack, self._labels_pack = self._pack(X_next, labels_next)

            self.labels[self.time_tag] = labels_curr

        self.__forgot_meanings()
        self.time_tag += 1

    def __forgot_meanings(self):
        if self.time_tag <= self.aging_index or self.aging_index == 0:
            self.cut.append(set())
            return

        # mapping
        index = 0
        for time_step in range(self.time_tag):
            n = self.labels[time_step].shape[0]
            self.labels[time_step] = self.labels_[index: index + n]
            index += n

        cut = set()
        for time_step in range(self.time_tag - 1 - self.aging_index, self.time_tag - 1):
            for label in np.unique(self.labels[time_step]):
                outdate = True
                for time_step_next in range(time_step + 1, self.time_tag):
                    if label in self.labels[time_step_next]:
                        outdate = False

                if outdate:
                    cut.add(label)
                    self.X_ = self.X_[self.labels_ != label]
                    self.to_remove.append(np.where(self.labels_ == label)[0])
                    self.labels_ = self.labels_[self.labels_ != label]
                    self._X_pack, self._labels_pack = self._pack(self.X_, self.labels_)

                    for t in range(self.time_tag - 1):
                        self.labels[t] = self.labels[t][self.labels[t] != label]
        self.cut.append(cut)
