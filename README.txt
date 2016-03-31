*pohmm* is an implementation of the partially observable hidden Markov model, a generalization of the hidden Markov model in which the underlying system state is partially observable through event metadata at each time step.

An application that motivates usage of such a model is keystroke biometrics where the user can be in either a passive or active hidden state at each time step, and the time between key presses depends on the hidden state. In addition, the hidden state depends on the key that was pressed; thus the keys are observed symbols that partially reveal the hidden state of the user.

For examples and documentation, see https://github.com/vmonaco/pohmm
