
import numpy as np


class RLS():
    def __init__(self, measurement, initial_prediction, compute_Jacobian_and_resid, forward_model, d_param=1e-50,
                 r_param=1e-50,
                 lower_constraint=-np.inf, upper_constraint=np.inf, num_iterations=5, search_direction='Levenberg-Marquardt', reg_type = 'Finite Difference'):

        self.reg_type = reg_type

        self.initial_prediction = np.copy(initial_prediction.reshape((len(initial_prediction), 1)))
        self.current_estimate = np.copy(self.initial_prediction)
        self.num_iterations = num_iterations

        self.compute_Jacobian_and_resid = compute_Jacobian_and_resid
        self.forward_model = forward_model
       

        self.verbose = True

        self.measurement = measurement.reshape((len(measurement),1))

        self.gamma = d_param
        self.lam = r_param
        self.J = np.zeros((len(measurement), len(self.initial_prediction)))
        self.residual = np.zeros((len(measurement), 1))

        self.lower_constraint = lower_constraint
        self.upper_constraint = upper_constraint

        self.search_direction = search_direction



        self.Jacobian_hist = np.full((len(self.measurement), len(self.initial_prediction),1), np.nan)
        self.residual_hist = np.full((len(self.measurement), 1,1), np.nan)
        self.current_est_hist = np.full((len(self.initial_prediction), 1,1), np.nan)

        self.regularisation_matrix()

        self.problem = 'non-linear'

        self.f0 = self.LS_function(self.initial_prediction)

        if self.search_direction == 'Levenberg-Marquardt' or self.search_direction == 'NL2SOL':
            import TRA.LM as TRA
            self.inversion_algorithm = TRA.Levenberg_Marquart(self.initial_prediction, self.f0, self.compute_hessian,
                                                          self.compute_gradient,
                                                          self.LS_function, d_param=self.gamma,
                                                          lower_constraint=self.lower_constraint,
                                                          upper_constraint=self.upper_constraint,
                                                          num_iterations=self.num_iterations)
        elif self.search_direction == "Powell's Dogleg":
            import TRA.PD as TRA
            self.inversion_algorithm = TRA.powells_dogleg(self.initial_prediction, self.f0, self.compute_hessian,
                                                          self.compute_gradient,
                                                          self.LS_function, constraint_region=self.gamma,
                                                          lower_constraint=self.lower_constraint,
                                                          upper_constraint=self.upper_constraint,
                                                          num_iterations=self.num_iterations)

    def LS_function(self, predicted):

        self.f_c = self.forward_model(predicted)
        residual = self.f_c - self.measurement

        penalty = np.matmul(self.RM, predicted - self.initial_prediction)
        f = 0.5 * np.linalg.norm(residual) ** 2 + 0.5 * self.lam * np.linalg.norm(penalty) ** 2

        return f

    def regularisation_matrix(self):
        if self.reg_type == 'Identity':
            RM = np.diag(np.ones((len(self.initial_prediction),1)))

        elif self.reg_type == 'Finite Difference':
            RM = np.zeros((len(self.initial_prediction), len(self.initial_prediction)))

            for i in range(0, 9):
                RM[i, i + 1] = 0.5
            for i in range(0, 9):
                RM[i + 1, i] = -0.5
            RM[0, 0] = -1
            RM[0, 1] = 1
            RM[-1, -1] = 1
            RM[-1, -2] = -1
        elif self.reg_type == 'None':
            RM = np.zeros((len(self.initial_prediction), len(self.initial_prediction)))


        self.RM = RM
        self.RM_square = np.matmul(np.transpose(self.RM), self.RM)

    def compute_hessian(self, current_estimate):

        if self.problem == 'non-linear':
            self.J, self.residual = self.compute_Jacobian_and_resid(current_estimate)


        self.Jacobian_hist = np.concatenate((self.Jacobian_hist, self.J.reshape((len(self.measurement), len(self.initial_prediction),1))),axis=2)
        self.residual_hist = np.concatenate((self.residual_hist, self.residual.reshape((len(self.measurement), 1,1))),axis=2)
        self.current_est_hist = np.concatenate((self.current_est_hist, current_estimate.reshape((len(self.initial_prediction), 1,1))),axis=2)


        self.JT = np.transpose(self.J)


        hess_regularised = np.matmul(self.JT, self.J) + self.lam * self.RM_square

        if self.search_direction == 'NL2SOL':
            length = len(self.residual_hist[0, 0, :])
            for i in range(1, length):
                residual_last = self.residual_hist[:, :, i - 1].reshape(np.shape(self.residual))
                residual_current = self.residual_hist[:, :, i].reshape(np.shape(self.residual))
                Jacobian_last = self.Jacobian_hist[:, :, i - 1].reshape(np.shape(self.J))
                Jacobian_current = self.Jacobian_hist[:, :, i].reshape(np.shape(self.J))
                current_estimate = self.current_est_hist[:, :, i].reshape(np.shape(self.initial_prediction))
                last_estimate = self.current_est_hist[:, :, i - 1].reshape(np.shape(self.initial_prediction))

                grad_unreg_current = np.matmul(np.transpose(Jacobian_current), residual_current)
                grad_unreg_last = np.matmul(np.transpose(Jacobian_last), residual_last)

                y_sharp = grad_unreg_current - np.matmul(np.transpose(Jacobian_last), residual_current)
                y = grad_unreg_current - grad_unreg_last
                s = current_estimate - last_estimate

                if (np.matmul(np.transpose(y), s)) == 0:
                    rho = 0
                else:
                    rho = 1 / (np.matmul(np.transpose(y), s))

                if any(np.isnan(rho)):
                    B = np.zeros((10, 10))

                else:
                    if np.linalg.norm(np.matmul(np.transpose(s), np.matmul(B_last, s))) == 0:
                        tau =1
                    else:
                        tau = np.min([1, np.linalg.norm(np.matmul(np.transpose(s), y_sharp)) / np.linalg.norm(
                            np.matmul(np.transpose(s), np.matmul(B_last, s)))])

                    B_last = tau * B_last;
                    term = (y_sharp - np.matmul(B_last, s)) * rho

                    B = B_last + np.matmul(term, np.transpose(y)) + np.matmul(y, np.transpose(term)) - \
                        rho*np.matmul(np.transpose(term), s)*np.matmul(y, np.transpose(y))


                B_last = B

            hess_regularised = np.matmul(self.JT, self.J)+B + self.lam * self.RM_square

        return hess_regularised

    def compute_gradient(self, current_estimate):

        self.JT = np.transpose(self.J)

        grad = np.matmul(self.JT, self.residual) + self.lam * np.matmul(self.RM_square,
                                                                        current_estimate - self.initial_prediction)

        return grad

    def compute_minimum(self):


        self.inversion_algorithm.current_estimate = self.current_estimate

        reconstructed = self.inversion_algorithm.optimisation_main()

        print(reconstructed)

        return reconstructed


