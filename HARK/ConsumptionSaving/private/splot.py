from HARK.utilities import plot_funcs
debugfile('/Volumes/Data/Papers/BufferStockTheory/BufferStockTheory-Latest/Code/Python/tmp.py',
          wdir='/Volumes/Data/Papers/BufferStockTheory/BufferStockTheory-Latest/Code/Python')
s = locals()['GICNrmFailsButGICRawHolds'].solution[0]
plot_funcs(s.Ex_m_tp1_minus_m_t, 0, 5, 100)
