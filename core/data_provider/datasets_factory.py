from core.data_provider import bair, action_mode1, action_mode2, radar_multi_gz, kth_action, mnist, kth_label,human,something,action,traffic_flow, radar_weather, mnist_pp, robonet

datasets_map = {
    'mnist': mnist,
    'action_RGB': kth_action,
    'action': action,
    'action_label': kth_label,
    'human' : human,
    "something" : something,
    "traffic_flow" : traffic_flow,
    'radar_weather' : radar_weather,
    'radar_multi_gz' : radar_multi_gz,
    'mnist_pp' : mnist_pp,
    'action_mode1' : action_mode1,
    'action_mode2' : action_mode2,
    'robonet' : robonet,
    'bair' : bair,
}


def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, is_training=True, specific_category=None, injection_action=None):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    if dataset_name == 'mnist':
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name + 'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle

    if 'action_rgb' in dataset_name:
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle


    if dataset_name == 'human':
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'channel': 3,
                       'input_data_type': 'float32',
                       'name': 'human'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        test_input_handle = input_handle.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle


    if dataset_name == "something":
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_param = {'paths': train_data_list,
                           'image_width': img_width,
                           'minibatch_size': batch_size,
                           'seq_length': seq_length,
                           'input_data_type': 'float32',
                           'name': dataset_name + ' iterator'}
            train_handle = datasets_map[dataset_name].DataProcess(train_param)
            train_input_handle = train_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle


    if 'action_mode1' in dataset_name :
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handles = input_handle.get_test_input_handle()
            #for (category, test_input_handle) in test_input_handles:
            #   test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handles
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle

    if 'action_mode2' in dataset_name:
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handles = input_handle.get_test_input_handle()
            # for (category, test_input_handle) in test_input_handles:
            #   test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handles
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle


    if 'action' in dataset_name or 'traffic_flow' in dataset_name:
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handles = input_handle.get_test_input_handle()
            #for (category, test_input_handle) in test_input_handles:
            #   test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handles
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle


    if dataset_name == 'radar_weather':
        input_param =  {'valid_data_paths':valid_data_paths,
                        'train_data_paths' :train_data_paths,
                        'small_data': False,
                        'image_width':img_width,
                        'minibatch_size': batch_size,
                        'time_revolution': 1,
                        'seq_length':seq_length,
                        'input_data_type': 'float32',
                        'name':'radar weather'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle = True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle = False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle = False)
            test_input_full_handle = input_handle.get_test_input_handle(full=True)
            test_input_full_handle.begin(do_shuffle = False)
            return test_input_handle, test_input_full_handle

    if dataset_name == 'radar_multi_gz':
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'image_width': img_width,
                            'image_height': img_width,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name + 'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'image_width': img_width,
                                 'image_height': img_width,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle


    if dataset_name == 'mnist_pp':
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'image_width': img_width,
                            'image_height': img_width,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name + 'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'image_width': img_width,
                                 'image_height': img_width,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=True)

            return train_input_handle, test_input_handle

    if dataset_name == 'robonet':
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'minibatch_size_test': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle(specific_category = specific_category)
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle(specific_category = specific_category)
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle(specific_category = specific_category)
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle

    if dataset_name == 'bair':
        test_input_param = {'valid_data_paths': valid_data_list,
                            'train_data_paths': train_data_list,
                            'batch_size': batch_size,
                            'image_width': img_width,
                            'image_height': img_width,
                            'seq_length': seq_length,
                            'injection_action': injection_action,
                            'input_data_type': 'float32',
                            'name': dataset_name + 'test iterator'}
        input_handle_test = datasets_map[dataset_name].DataProcess(test_input_param)
        test_input_handle = input_handle_test.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'valid_data_paths': valid_data_list,
                                 'train_data_paths': train_data_list,
                                 'image_width': img_width,
                                 'image_height': img_width,
                                 'batch_size': batch_size,
                                 'seq_length': seq_length,
                                 'injection_action': injection_action,
                                 'input_data_type': 'float32',
                                 'name': dataset_name + ' train iterator'}
            input_handle_train = datasets_map[dataset_name].DataProcess(train_input_param)
            train_input_handle = input_handle_train.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle

