import run_analyzer as ra
meson_run22 = ra.run_2p("../../data_from_scp/tm_mesons_run22.mesons.dat",corr_to_mu_ratio=3)
meson_run22.mass_extraction(show=True,l_cut=5,max_chi2=1.0,zoom_out=2)
