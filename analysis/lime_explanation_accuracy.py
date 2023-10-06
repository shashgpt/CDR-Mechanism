import numpy as np
import math


class Lime_explanation_accuracy(object):

    def getStats(self, dist):
        m = np.mean(dist)
        std = np.std(dist)
        h = 1.96*std/math.sqrt(len(dist))
        
        m = round(m, 3)
        h = round(h, 3)
        return m, h

    def calculate_mean_std(self, base_lime_corrects, mask_lime_corrects, mask_contrast_lime_corrects):

        distributions_one_rule = [base_lime_corrects['one_rule'], mask_lime_corrects['one_rule'], mask_contrast_lime_corrects['one_rule']]
        distributions_one_rule_contrast = [base_lime_corrects['one_rule_contrast'], mask_lime_corrects['one_rule_contrast'], mask_contrast_lime_corrects['one_rule_contrast']]
        distributions_one_rule_no_contrast = [base_lime_corrects['one_rule_no_contrast'], mask_lime_corrects['one_rule_no_contrast'], mask_contrast_lime_corrects['one_rule_no_contrast']]
        
        distributions_a_but_b = [base_lime_corrects['a_but_b'], mask_lime_corrects['a_but_b'], mask_contrast_lime_corrects['a_but_b']]
        distributions_a_but_b_contrast = [base_lime_corrects['a_but_b_contrast'], mask_lime_corrects['a_but_b_contrast'], mask_contrast_lime_corrects['a_but_b_contrast']]
        distributions_a_but_b_no_contrast = [base_lime_corrects['a_but_b_no_contrast'], mask_lime_corrects['a_but_b_no_contrast'], mask_contrast_lime_corrects['a_but_b_no_contrast']]
        
        distributions_a_yet_b = [base_lime_corrects['a_yet_b'], mask_lime_corrects['a_yet_b'], mask_contrast_lime_corrects['a_yet_b']]
        distributions_a_yet_b_contrast = [base_lime_corrects['a_yet_b_contrast'], mask_lime_corrects['a_yet_b_contrast'], mask_contrast_lime_corrects['a_yet_b_contrast']]
        distributions_a_yet_b_no_contrast = [base_lime_corrects['a_yet_b_no_contrast'], mask_lime_corrects['a_yet_b_no_contrast'], mask_contrast_lime_corrects['a_yet_b_no_contrast']]

        distributions_a_though_b = [base_lime_corrects['a_though_b'], mask_lime_corrects['a_though_b'], mask_contrast_lime_corrects['a_though_b']]
        distributions_a_though_b_contrast = [base_lime_corrects['a_though_b_contrast'], mask_lime_corrects['a_though_b_contrast'], mask_contrast_lime_corrects['a_though_b_contrast']]
        distributions_a_though_b_no_contrast = [base_lime_corrects['a_though_b_no_contrast'], mask_lime_corrects['a_though_b_no_contrast'], mask_contrast_lime_corrects['a_though_b_no_contrast']]

        distributions_a_while_b = [base_lime_corrects['a_while_b'], mask_lime_corrects['a_while_b'], mask_contrast_lime_corrects['a_while_b']]
        distributions_a_while_b_contrast = [base_lime_corrects['a_while_b_contrast'], mask_lime_corrects['a_while_b_contrast'], mask_contrast_lime_corrects['a_while_b_contrast']]
        distributions_a_while_b_no_contrast = [base_lime_corrects['a_while_b_no_contrast'], mask_lime_corrects['a_while_b_no_contrast'], mask_contrast_lime_corrects['a_while_b_no_contrast']]
        
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

        data_means_std = {"base_model":{"one_rule":[correct_means_one_rule[0], correct_std_one_rule[0]],
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
                                        "a_while_b_no_contrast":[correct_means_a_while_b_no_contrast[0], correct_std_a_while_b_no_contrast[0]]},

                            "mask_model":{"one_rule":[correct_means_one_rule[1], correct_std_one_rule[1]],
                                        "one_rule_contrast":[correct_means_one_rule_contrast[1], correct_std_one_rule_contrast[1]],
                                        "one_rule_no_contrast":[correct_means_one_rule_no_contrast[1], correct_std_one_rule_no_contrast[1]],
                                        "a_but_b":[correct_means_a_but_b[1], correct_std_a_but_b[1]],
                                        "a_but_b_contrast":[correct_means_a_but_b_contrast[1], correct_std_a_but_b_contrast[1]],
                                        "a_but_b_no_contrast":[correct_means_a_but_b_no_contrast[1], correct_std_a_but_b_no_contrast[1]],
                                        "a_yet_b":[correct_means_a_yet_b[1], correct_std_a_yet_b[1]],
                                        "a_yet_b_contrast":[correct_means_a_yet_b_contrast[1], correct_std_a_yet_b_contrast[1]],
                                        "a_yet_b_no_contrast":[correct_means_a_yet_b_no_contrast[1], correct_std_a_yet_b_no_contrast[1]],
                                        "a_though_b":[correct_means_a_though_b[1], correct_std_a_though_b[1]],
                                        "a_though_b_contrast":[correct_means_a_though_b_contrast[1], correct_std_a_though_b_contrast[1]],
                                        "a_though_b_no_contrast":[correct_means_a_though_b_no_contrast[1], correct_std_a_though_b_no_contrast[1]],
                                        "a_while_b":[correct_means_a_while_b[1], correct_std_a_while_b[1]],
                                        "a_while_b_contrast":[correct_means_a_while_b_contrast[1], correct_std_a_while_b_contrast[1]],
                                        "a_while_b_no_contrast":[correct_means_a_while_b_no_contrast[1], correct_std_a_while_b_no_contrast[1]]},

                            "mask_contrast_model":{"one_rule":[correct_means_one_rule[2], correct_std_one_rule[2]],
                                        "one_rule_contrast":[correct_means_one_rule_contrast[2], correct_std_one_rule_contrast[2]],
                                        "one_rule_no_contrast":[correct_means_one_rule_no_contrast[2], correct_std_one_rule_no_contrast[2]],
                                        "a_but_b":[correct_means_a_but_b[2], correct_std_a_but_b[2]],
                                        "a_but_b_contrast":[correct_means_a_but_b_contrast[2], correct_std_a_but_b_contrast[2]],
                                        "a_but_b_no_contrast":[correct_means_a_but_b_no_contrast[2], correct_std_a_but_b_no_contrast[2]],
                                        "a_yet_b":[correct_means_a_yet_b[2], correct_std_a_yet_b[2]],
                                        "a_yet_b_contrast":[correct_means_a_yet_b_contrast[2], correct_std_a_yet_b_contrast[2]],
                                        "a_yet_b_no_contrast":[correct_means_a_yet_b_no_contrast[2], correct_std_a_yet_b_no_contrast[2]],
                                        "a_though_b":[correct_means_a_though_b[2], correct_std_a_though_b[2]],
                                        "a_though_b_contrast":[correct_means_a_though_b_contrast[2], correct_std_a_though_b_contrast[2]],
                                        "a_though_b_no_contrast":[correct_means_a_though_b_no_contrast[2], correct_std_a_though_b_no_contrast[2]],
                                        "a_while_b":[correct_means_a_while_b[2], correct_std_a_while_b[2]],
                                        "a_while_b_contrast":[correct_means_a_while_b_contrast[2], correct_std_a_while_b_contrast[2]],
                                        "a_while_b_no_contrast":[correct_means_a_while_b_no_contrast[2], correct_std_a_while_b_no_contrast[2]]}
                        }

        return data_means_std