import os

def create_experiment_folder(base_dir):
    # Получаем список всех папок в базовой директории
    existing_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # Фильтруем папки, которые начинаются с 'exp_'
    exp_folders = [f for f in existing_folders if f.startswith('exp_')]

    # Извлекаем номера из имен папок и находим максимальный номер
    if exp_folders:
        numbers = [int(f.split('_')[1]) for f in exp_folders]
        max_number = max(numbers)
    else:
        max_number = -1

    # Определяем следующий номер
    next_number = max_number + 1

    # Создаем имя новой папки
    new_folder_name = f'exp_{next_number}'

    # Создаем новую папку
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    print(f'Created folder: {new_folder_path}')
    return new_folder_path
