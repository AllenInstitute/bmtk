class PopTypes:
    """Essentially an enum to store the type/group of each cell. It's faster and more robust than doing multiple string
    comparisons.
    """
    Internal = 0
    External = 1
    Other = 2  # should never really get here

    @staticmethod
    def len():
        return 3


class PropertySchema(object):
    #######################################
    # For nodes/cells properties
    #######################################
    def get_pop_type(self, pop_params):
        model_type = pop_params['model_type'].lower()
        if model_type == 'virtual' or model_type == 'external':
            return PopTypes.External
        elif model_type == 'internal':
            return PopTypes.Internal
        else:
            return PopTypes.Unknown

    def get_params_column(self):
        raise NotImplementedError()
