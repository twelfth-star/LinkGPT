# You may modify this file to add more prompts for different datasets.

copurchasing_prompts = {
    'yn':{
        'task_desc': '',
        'source_node_intro': 'This is a product on Amazon:\n',
        'candidate_target_node_intro': 'This is another product:\n',
        'connection_question': 'Is this product also bought by the same user?\n',
    },
    'np':{
        'task_desc': '',
        'source_node_intro': 'This is a product on Amazon:\n',
        'question': 'What products are also bought by the same user?\n',
    }
}

citation_prompts = {
    'yn':{
        'task_desc': '',
        'source_node_intro': 'This is the source paper:\n',
        'candidate_target_node_intro': 'This is another paper:\n',
        'connection_question': 'Is this paper cited by the source paper?\n',
    },
    'np':{
        'task_desc': '',
        'source_node_intro': 'This is the source paper:\n',
        'question': 'What papers are cited by the source paper?\n',
    }
}

general_prompts = {
    'yn':{
        'task_desc': "Determine whether there is a link between the source node and the candidate nodes.\n",
        'source_node_intro': 'Source node:\n',
        'candidate_target_node_intro': 'Candidate target node:\n',
        'connection_question': 'Is this connected to the source node?\n',
    },
    'np':{
        'task_desc': '',
        'source_node_intro': 'Source node:\n',
        'question': 'What neighbors does this node have?\n',
    }
}

def get_prompts(dataset_name: str, task_name: str, allow_general_prompts: bool = True):
    if dataset_name.startswith('amazon'):
        prompts = copurchasing_prompts
    elif dataset_name.startswith('mag'):
        prompts = citation_prompts
    else:
        if allow_general_prompts:
            print(f"Warning: Dataset {dataset_name} not recognized. Using general prompts.")
            prompts = general_prompts
        else:
            raise ValueError(f"Dataset {dataset_name} not recognized. Must be one of 'amazon' or 'mag'.")
    
    if task_name in {'yn', 'np'}:
        return prompts[task_name]
    else:
        raise ValueError(f"Task name {task_name} not recognized. Must be one of 'yn' or 'np'.")