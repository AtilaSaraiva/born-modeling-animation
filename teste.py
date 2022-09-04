#importing libraries
import numpy as np
# Configure model presets
from examples.seismic import demo_model, TimeAxis
from examples.seismic import plot_velocity, plot_perturbation
from devito import gaussian_smooth
from examples.seismic import AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():

    nbl = int(0.3*201)
    model =  demo_model('layers-isotropic', origin=(0., 0.), shape=(201, 201),
                        spacing=(10., 10.), nbl=nbl, nlayers=2)

    filter_sigma = (1, 1)
    nshots = 1
    nreceivers = 201
    t0 = 0.
    tn = 1500.  # Simulation last 1 second (1000 ms)
    f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
    dt = model.critical_dt
    time_range = TimeAxis(start=t0,stop=tn,step=dt)

    # Create initial model and smooth the boundaries
    model0 = demo_model('layers-isotropic', origin=(0., 0.), shape=(201, 201),
                        spacing=(10., 10.), nbl=nbl, nlayers=2)
    gaussian_smooth(model0.vp, sigma=filter_sigma)

    # Plot the true and initial model and the perturbation between them
    # plot_velocity(model)
    # plot_velocity(model0)
    # plot_perturbation(model0, model)

    # First, position source centrally in all dimensions, then set depth
    src_coordinates = np.empty((1, 2))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = 20.  # Depth is 20m

    # Define acquisition geometry: receivers
    # Initialize receivers for synthetic and imaging data
    rec_coordinates = np.empty((nreceivers, 2))
    rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
    rec_coordinates[:, 1] = 30.

    # Geometry
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, f0=f0, src_type='Ricker')


    solver = AcousticWaveSolver(model, geometry, space_order=4)
    true_d , _, _ = solver.forward(vp=model.vp)
    smooth_d, u0, _ = solver.forward(vp=model0.vp, save=True)


    def plotField(u0,u1, file):
        fig, (ax1, ax2) = plt.subplots(1,2)
        #creating a subplot
        # ax1 = fig.add_subplot(1,1,1)
        # ax2 = fig.add_subplot(1,2,1)
        vmin = -1
        vmax = +1
        aux = np.zeros((201,201))
        aux[0

        def animate(i):
            ax1.clear()
            data = u0.data[i,nbl:-nbl,nbl:-nbl]
            ax1.set_title("tempo = %i"%i)
            ax1.set_xlabel("x")
            ax1.set_ylabel("z")
            ax1.imshow(np.transpose(data),vmin=vmin, vmax=vmax, cmap="seismic")
            ax2.clear()
            if i < time_range.num/2:
                data = np.zeros((201, 201))
            else:
                i_aux = i - int(time_range.num/2)
                data = u1.data[i_aux,nbl:-nbl,nbl:-nbl] * d
            ax2.set_title("tempo = %i"%i)
            ax2.set_xlabel("x")
            ax2.set_ylabel("z")
            ax2.imshow(np.transpose(data),vmin=vmin, vmax=vmax, cmap="seismic")

        # plot a cada 20 snaps
        snap_interval = 10
        ani = animation.FuncAnimation(fig, animate, frames=range(0,u0.data.shape[0],snap_interval),
                                      interval=100, repeat=False)
        ani.save(file, writer='imagemagick', fps=20)

    # plotField(u0, "ondas.gif")

    # First, position source centrally in all dimensions, then set depth
    src_coordinates = np.empty((1, 2))
    src_coordinates[0, :] = np.array(model.domain_size) * .5

    # Geometry
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn/2, f0=f0, src_type='Ricker')

    solver = AcousticWaveSolver(model, geometry, space_order=4)
    dm_true = (solver.model.vp.data**(-2) - model0.vp.data**(-2))
    smooth_d, u1, _ = solver.forward(vp=model0.vp, save=True)
    plotField(u0,u1,dm_true "ondas1.gif")

if __name__ == "__main__":
    main()
