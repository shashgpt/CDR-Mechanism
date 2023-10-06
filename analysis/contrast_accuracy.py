import numpy as np
import math

class Contrast_accuracy(object):

    def getStats(self, dist):
        m = np.mean(dist)
        std = np.std(dist)
        h = 1.96*std/math.sqrt(len(dist))
        
        m = round(m, 3)
        h = round(h, 3)
        return m, h

    def calculate_mean_std(self, mask_contrast_contrast_corrects):

        distributions_overall = [mask_contrast_contrast_corrects['overall']]

        distributions_no_rule = [mask_contrast_contrast_corrects['no_rule']]

        distributions_one_rule = [mask_contrast_contrast_corrects['one_rule']]
        distributions_one_rule_contrast = [mask_contrast_contrast_corrects['one_rule_contrast']]
        distributions_one_rule_no_contrast = [mask_contrast_contrast_corrects['one_rule_no_contrast']]

        distributions_a_but_b = [mask_contrast_contrast_corrects['a_but_b']]
        distributions_a_but_b_contrast = [mask_contrast_contrast_corrects['a_but_b_contrast']]
        distributions_a_but_b_no_contrast = [mask_contrast_contrast_corrects['a_but_b_no_contrast']]

        distributions_a_yet_b = [mask_contrast_contrast_corrects['a_yet_b']]
        distributions_a_yet_b_contrast = [mask_contrast_contrast_corrects['a_yet_b_contrast']]
        distributions_a_yet_b_no_contrast = [mask_contrast_contrast_corrects['a_yet_b_no_contrast']]

        distributions_a_though_b = [mask_contrast_contrast_corrects['a_though_b']]
        distributions_a_though_b_contrast = [mask_contrast_contrast_corrects['a_though_b_contrast']]
        distributions_a_though_b_no_contrast = [mask_contrast_contrast_corrects['a_though_b_no_contrast']]

        distributions_a_while_b = [mask_contrast_contrast_corrects['a_while_b']]
        distributions_a_while_b_contrast = [mask_contrast_contrast_corrects['a_while_b_contrast']]
        distributions_a_while_b_no_contrast = [mask_contrast_contrast_corrects['a_while_b_no_contrast']]

        # Corrects means and std
        correct_means_overall = [self.getStats(x)[0] for x in distributions_overall]
        correct_std_overall = [self.getStats(x)[1] for x in distributions_overall]

        correct_means_no_rule = [self.getStats(x)[0] for x in distributions_no_rule]
        correct_std_no_rule = [self.getStats(x)[1] for x in distributions_no_rule]

        correct_means_one_rule = [self.getStats(x)[0] for x in distributions_one_rule]
        correct_std_one_rule = [self.getStats(x)[1] for x in distributions_one_rule]

        correct_means_one_rule_contrast = [self.getStats(x)[0] for x in distributions_one_rule_contrast]
        correct_std_one_rule_contrast = [self.getStats(x)[1] for x in distributions_one_rule_contrast]

        correct_means_one_rule_no_contrast = [self.getStats(x)[0] for x in distributions_one_rule_no_contrast]
        correct_std_one_rule_no_contrast = [self.getStats(x)[1] for x in distributions_one_rule_no_contrast]

        correct_means_a_but_b = [self.getStats(x)[0] for x in distributions_a_but_b]
        correct_std_a_but_b = [self.getStats(x)[1] for x in distributions_a_but_b]
        correct_means_a_but_b_contrast = [self.getStats(x)[0] for x in distributions_a_but_b_contrast]
        correct_std_a_but_b_contrast = [self.getStats(x)[1] for x in distributions_a_but_b_contrast]
        correct_means_a_but_b_no_contrast = [self.getStats(x)[0] for x in distributions_a_but_b_no_contrast]
        correct_std_a_but_b_no_contrast = [self.getStats(x)[1] for x in distributions_a_but_b_no_contrast]

        correct_means_a_yet_b = [self.getStats(x)[0] for x in distributions_a_yet_b]
        correct_std_a_yet_b = [self.getStats(x)[1] for x in distributions_a_yet_b]
        correct_means_a_yet_b_contrast = [self.getStats(x)[0] for x in distributions_a_yet_b_contrast]
        correct_std_a_yet_b_contrast = [self.getStats(x)[1] for x in distributions_a_yet_b_contrast]
        correct_means_a_yet_b_no_contrast = [self.getStats(x)[0] for x in distributions_a_yet_b_no_contrast]
        correct_std_a_yet_b_no_contrast = [self.getStats(x)[1] for x in distributions_a_yet_b_no_contrast]

        correct_means_a_though_b = [self.getStats(x)[0] for x in distributions_a_though_b]
        correct_std_a_though_b = [self.getStats(x)[1] for x in distributions_a_though_b]
        correct_means_a_though_b_contrast = [self.getStats(x)[0] for x in distributions_a_though_b_contrast]
        correct_std_a_though_b_contrast = [self.getStats(x)[1] for x in distributions_a_though_b_contrast]
        correct_means_a_though_b_no_contrast = [self.getStats(x)[0] for x in distributions_a_though_b_no_contrast]
        correct_std_a_though_b_no_contrast = [self.getStats(x)[1] for x in distributions_a_though_b_no_contrast]

        correct_means_a_while_b = [self.getStats(x)[0] for x in distributions_a_while_b]
        correct_std_a_while_b = [self.getStats(x)[1] for x in distributions_a_while_b]
        correct_means_a_while_b_contrast = [self.getStats(x)[0] for x in distributions_a_while_b_contrast]
        correct_std_a_while_b_contrast = [self.getStats(x)[1] for x in distributions_a_while_b_contrast]
        correct_means_a_while_b_no_contrast = [self.getStats(x)[0] for x in distributions_a_while_b_no_contrast]
        correct_std_a_while_b_no_contrast = [self.getStats(x)[1] for x in distributions_a_while_b_no_contrast]

        # List of means
        data_means_std = {"mask_contrast_model":{"overall":[correct_means_overall[0], correct_std_overall[0]], 
                                                "no_rule":[correct_means_no_rule[0], correct_std_no_rule[0]], 
                                                "one_rule":[correct_means_one_rule[0], correct_std_one_rule[0]],
                                                "one_rule_contrast":[correct_means_one_rule_contrast[0], correct_std_one_rule_contrast[0]],
                                                "one_rule_no_contrast":[correct_means_one_rule_no_contrast[0], correct_std_one_rule_no_contrast[0]],
                                                "a_but_b":[correct_means_a_but_b[0], correct_std_a_but_b[0]],
                                                "a_but_b_contrast":[correct_means_a_but_b_contrast[0], correct_std_a_but_b_contrast[0]],
                                                "a_but_b_no_contrast":[correct_means_a_but_b_no_contrast[0], correct_std_a_but_b_no_contrast[0]],
                                                "a_yet_b":[correct_means_a_yet_b[0], correct_std_a_yet_b[0]],
                                                "a_yet_b_contrast":[correct_means_a_yet_b_contrast[0], correct_std_a_yet_b_contrast[0]],
                                                "a_yet_b_no_contrast":[correct_means_a_yet_b_no_contrast[0], correct_std_a_yet_b_no_contrast[0]],
                                                "a_though_b":[correct_means_a_though_b[0], correct_std_a_though_b[0]],
                                                "a_though_b_contrast":[correct_means_a_though_b_contrast[0], correct_std_a_though_b_contrast[0]],
                                                "a_though_b_no_contrast":[correct_means_a_though_b_no_contrast[0], correct_std_a_though_b_no_contrast[0]],
                                                "a_while_b":[correct_means_a_while_b[0], correct_std_a_while_b[0]],
                                                "a_while_b_contrast":[correct_means_a_while_b_contrast[0], correct_std_a_while_b_contrast[0]],
                                                "a_while_b_no_contrast":[correct_means_a_while_b_no_contrast[0], correct_std_a_while_b_no_contrast[0]]}
                        }
        
        return data_means_std