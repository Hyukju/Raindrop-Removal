import importlib

def find_model(model_name, phase='train', dataset_dir=None, use_transform=True, learning_rate=0.002):
    '''
    model 폴더 내부에 존재하는 _model.py와 _loader.py를 찾고 model_name을 기준으로 찾고 
    model과 dataset을 instance 시킴 

    '''
    
    model, dataset = None, None 
    
    model_dir = 'model.'
    model_module_name = model_dir + model_name.lower() + '_model'

    print('searching... ', model_module_name)
    # model_module_name.py 모듈을 import함 
    model_module = importlib.import_module(model_module_name)
    # model_module_name.py 모듈 내부의 class 이름  
    target_model_name = 'Model'

    # 클래스를 찾음 
    for name, cls in model_module.__dict__.items():
        if name.lower() == target_model_name.lower():
            # 클래스를 instance 시킴 
            model = cls

    # 클래스가 존재하지 않을 경우 경고
    if model is None:
        print(f'{model_module_name}.{target_model_name} 가 존재하지 않습니다.')
    else:
        print(f'{model_module_name} is created')
        model = model(phase=phase, in_channels=3, out_channels=3, nker=64, lr=learning_rate)

    # 데이터 로더 (내용 동일)
    if phase == 'train':        
        loader_module_name = model_dir + model_name.lower() + '_loader'
        print('searching... ', loader_module_name)    
        loader_module = importlib.import_module(loader_module_name)
        target_loader_name = 'ModelDataset'
        
        for name, cls in loader_module.__dict__.items():
            if name.lower() == target_loader_name.lower():
                dataset = cls

        if dataset is None:
            print(f'{loader_module_name}.{target_loader_name} 가 존재하지 않습니다.')
        else:
            print(f'{loader_module_name} is created')
            dataset = dataset(dataset_dir, use_transform=use_transform)
    return  model, dataset
