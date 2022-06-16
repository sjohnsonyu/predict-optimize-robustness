import tensorflow as tf
import numpy as np
import tqdm

def projection(perturbed_label, label, budget, norm=2):
    perturbed_label = tf.clip_by_value(perturbed_label, 0, 1)
    perturbation = tf.clip_by_norm(perturbed_label - label, budget)
    return label + perturbation

def perturbation_df(ope_simulator, w, K, label, budget, norm=2, step_size=1e-2):
    perturbed_label = tf.Variable(label)
    adversarial_iterations = 1000
    for _ in range(adversarial_iterations):
        with tf.GradientTape() as tape:
            tf.watch(perturbed_label)
            perturbed_reward = ope_simulator.perturbation(w, K, perturbed_label)
            
        dreward_dperturbation = tape.gradient(perturbed_reward, perturbed_label)
        perturbed_label -= dreward_dperturbation * step_size
        perturbed_label = projection(perturbed_label, label, budget)
        del tape

    return perturbed_label
