from bmtk.utils.reports import CompartmentReport

report = CompartmentReport('tmp_report.h5', mode='r')
print(report.populations)
for pop in report.populations:
    #print(report[pop].node_ids())
    #print(report[pop].data(0))
    #print(report[pop].element_ids())
    #print(report[pop].n_elements(0))
    #print(report[pop].n_elements(1))
    #print(report[pop].n_elements(2))
    #print(report[pop].n_elements(3))
    #print(report[pop].t_start())
    #print(len(report[pop].time_trace()))
    #print(report[pop].n_steps())
    #print(report[pop].data(1))
    print(report[pop].data(1, sections='origin').shape)


