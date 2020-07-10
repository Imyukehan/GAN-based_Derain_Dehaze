import os
import importlib

def print_parameters(model, model_name):
    print(model_name, '-', end=' ')
    print(sum([len(p) for p in model.parameters()]))


def main():
    models = [os.path.splitext(f)[0] for f in os.listdir(os.path.join('CODE', 'models'))]

    for model_name in models:
        module = importlib.import_module('CODE.models.%s' %model_name)
        if hasattr(module, 'Generator'):
            print_parameters(module.Generator(), 'Generator [{}]'.format(model_name))
        #elif hasattr(module, 'GeneratorUNet'):
        #    print_parameters(module.GeneratorUNet(), 'Generator [{}]'.format(model_name))
        if hasattr(module, 'Discriminator'):
            print_parameters(module.Discriminator(), 'Discriminator [{}]'.format(model_name))

if __name__ == '__main__':
    main()