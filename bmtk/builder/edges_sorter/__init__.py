from .memory_sorter import quicksort_edges
from .merge_sorter import external_merge_sort


def sort_edges(input_edges_path, output_edges_path, edges_population, sort_by, sort_model_properties=True,
               sort_on_disk=False, **sorter_args):
    if not sort_on_disk:
        quicksort_edges(
            input_edges_path=input_edges_path,
            output_edges_path=output_edges_path,
            edges_population=edges_population,
            sort_by=sort_by,
            sort_model_properties=sort_model_properties,
            **sorter_args
        )
    else:
        external_merge_sort(
            input_edges_path=input_edges_path,
            output_edges_path=output_edges_path,
            edges_population=edges_population,
            sort_by=sort_by,
            sort_model_properties=sort_model_properties,
            **sorter_args
        )
