import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from argparse import ArgumentParser

usage = "usage: python %(prog)s <output directory> [optional 2nd output directory] [other options]"
parser = ArgumentParser(usage=usage)

parser.add_argument(
    "model_dir", type=str, action="store", help="directory containing outputs to plot"
)
parser.add_argument(
    "reference_dir",
    type=str,
    action="store",
    help="directory containing outputs to compare to",
    nargs="?",
)
args = parser.parse_args()

rainbow_colorsheme = [
    "#ec1b8c",
    "#a6228e",
    "#20419a",
    "#0085cc",
    "#03aeef",
    "#03aa4f",
    "#c8da2c",
    "#fff200",
    "#f99e1c",
    "#ed1c24",
]

np.set_printoptions(precision=14)

##################
# Data Wrangling #
##################

datafiles = [
    r"outstate_0.nc",
    r"outstate_1.nc",
    r"outstate_2.nc",
    r"outstate_3.nc",
    r"outstate_4.nc",
    r"outstate_5.nc",
]

surface_pressure_plots = []
surface_temperature_plots = []

vardict = {}

for filename in datafiles:
    fname = args.model_dir + filename
    ncfile = Dataset(fname, "r")
    nc_attrs = ncfile.ncattrs()
    nc_dims = [dim for dim in ncfile.dimensions]  # list of nc dimensions
    nc_vars = [var for var in ncfile.variables]  # list of nc variables

    surface_pressure = ncfile.variables["surface_pressure"][:].data / 100.0  # convert to hPa
    temperature = ncfile.variables["air_temperature"][:].data

    surface_temperature = temperature[-1, :, :]  # temperature at bottom

    if args.reference_dir:
        fname2 = args.reference_dir + f
        ncf2 = Dataset(fname2, "r")
        surface_pressure2 = ncf2.variables["surface_pressure"][:].data / 100.0  # convert to hPa
        temperature2 = ncf2.variables["air_temperature"][:].data
        surface_temperature2 = temperature2[-1, :, :]  # field at 850 hPa

        surface_pressure_diff = (surface_pressure - surface_pressure2) / surface_pressure2
        temperature_diff = (surface_temperature - surface_temperature2) / surface_temperature2

        surface_pressure_plots.append(surface_pressure_diff)
        surface_temperature_plots.append(temperature_diff)

        # savin' variables
        for var in nc_vars:
            if ("time" not in var):
                if var in ncf2.variables.keys():
                    if var not in vardict.keys():
                        vardict[var] = []
                    field1 = ncfile[var][:]
                    field2 = ncf2[var][:]
                    vardict[var].append((field1 - field2) / field2)

    else:
        surface_pressure_plots.append(surface_pressure)
        surface_temperature_plots.append(surface_temperature)


####################
# Doin' some stats #
####################
if args.reference_dir:
    pminmaxes = []
    pmeans = []
    p_lnorm1 = []
    p_lnorm2 = []
    p_lnorminf = []

    tminmaxes = []
    tmeans = []
    t_lnorm1 = []
    t_lnorm2 = []
    t_lnorminf = []

    vminmaxes = []
    vmeans = []
    variable_lnorm1 = []
    variable_lnorm2 = []
    variable_lnorminf = []

    p = np.array(surface_pressure_plots)
    tp = np.array(surface_temperature_plots)
    pres_date = np.concatenate(
        (p[0, :, :], p[1, :, :], p[2, :, :], p[3, :, :], p[4, :, :], p[5, :, :]), axis=0
    )
    pminmaxes.append([pres_date.min(), pres_date.max()])
    pmeans.append(pres_date.mean())
    p_lnorm1.append(np.linalg.norm(pres_date.flatten(), 1))
    p_lnorm2.append(np.linalg.norm(pres_date.flatten(), 2))
    p_lnorminf.append(np.linalg.norm(pres_date.flatten(), np.inf))

    surface_temperature_date = np.concatenate(
        (tp[0, :, :], tp[1, :, :], tp[2, :, :], tp[3, :, :], tp[4, :, :], tp[5, :, :]),
        axis=0,
    )
    tminmaxes.append([surface_temperature_date.min(), surface_temperature_date.max()])
    tmeans.append(surface_temperature_date.mean())
    t_lnorm1.append(np.linalg.norm(surface_temperature_date.flatten(), 1))
    t_lnorm2.append(np.linalg.norm(surface_temperature_date.flatten(), 2))
    t_lnorminf.append(np.linalg.norm(surface_temperature_date.flatten(), np.inf))

    print(pmeans)
    print(tmeans)

    for var in vardict.keys():
        var_tiles = np.concatenate(
            (
                vardict[var][0],
                vardict[var][1],
                vardict[var][2],
                vardict[var][3],
                vardict[var][4],
                vardict[var][5],
            ),
            axis=0,
        ).flatten()
        vminmaxes.append([var_tiles.min(), var_tiles.max()])
        vmeans.append(var_tiles.mean())
        variable_lnorm1.append(np.linalg.norm(var_tiles.flatten(), 1))
        variable_lnorm2.append(np.linalg.norm(var_tiles.flatten(), 2))
        variable_lnorminf.append(np.linalg.norm(var_tiles.flatten(), np.inf))   
        print(var, np.mean(np.abs(var_tiles.flatten())))

################
# Making Plots #
################

plotdir = "raw_state"

# Stuff for to make plots more prettier
axwidth = 3
axlength = 12
fontsize = 20
linewidth = 6
labelsize = 20

plt.rc("text.latex", preamble=r"\boldmath")
plt.rc("text", usetex=True)

# Plot params:
minlon = 1
maxlon = 48
minlat = 1
maxlat = 48

colorbar_array = [0.1, 0.05, 0.8, 0.05]
bottom_adjust = 0.15

pressure_levels = 5
temperature_levels = 5

if args.reference_dir:
    post = "diff"
else:
    post = "range"

tilestrs = [
    r"$\mathrm{Tile\ 1}$",
    r"$\mathrm{Tile\ 2}$",
    r"$\mathrm{Tile\ 3}$",
    r"$\mathrm{Tile\ 4}$",
    r"$\mathrm{Tile\ 5}$",
    r"$\mathrm{Tile\ 6}$",
]

# Pressure plots
fig1, axs1 = plt.subplots(3, 2)
levs = []
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        if k == 0:
            cs = axs1[ii, jj].contourf(np.log10(np.abs(surface_pressure_plots[k])), pressure_levels)
            levs = cs.levels
        else:
            cs = axs1[ii, jj].contourf(np.log10(np.abs(surface_pressure_plots[k])), levs)
        axs1[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig1.subplots_adjust(bottom=bottom_adjust)
cax = fig1.add_axes(colorbar_array)
cbar = fig1.colorbar(cs, cax=cax, orientation="horizontal")

fig1.suptitle(r"$\mathrm{Log_{10}\ Surface\ Pressure}$")

plt.savefig("{0}/tile_pressure0_{1}.png".format(plotdir, post))

# Surface Temperature plots
fig2, axs2 = plt.subplots(3, 2)
for ii in range(3):
    for jj in range(2):
        k = 2 * ii + jj
        if k == 0:
            cs = axs2[ii, jj].contourf(np.log10(np.abs(surface_temperature_plots[k])), temperature_levels)
            levs = cs.levels
        else:
            cs = axs2[ii, jj].contourf(np.log10(np.abs(surface_temperature_plots[k])), levs)
        axs2[ii, jj].annotate(tilestrs[k], (2, 2), textcoords="data", size=9)

fig2.subplots_adjust(bottom=bottom_adjust)
cax = fig2.add_axes(colorbar_array)
cbar = fig2.colorbar(cs, cax=cax, orientation="horizontal")

fig2.suptitle(r"$\mathrm{Log_{10}\ Bottom\ Temperature}$")\

plt.savefig("{0}/tile_bot_temperature0_{1}.png".format(plotdir, post))
