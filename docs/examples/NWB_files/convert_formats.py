from bmtk.utils.reports.spike_trains import SpikeTrains, sort_order

st = SpikeTrains.from_nwb('lgn_spikes.nwb', population='lgn')
#st.to_csv('lgn_spikes.csv')
#st.to_csv('lgn_spikes.nopop.csv', include_population=False)
st.to_sonata('lgn_spikes.h5', sort_order=sort_order.by_id)