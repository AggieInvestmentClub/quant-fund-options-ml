"""Build CRR and LR/OPRA native extensions on the destination machine."""
from __future__ import annotations
import argparse,os,shlex,shutil,subprocess,sys,sysconfig
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['all','crr','lr'],default='all')
    p.add_argument('--jobs',type=int,choices=range(1,5),default=4)
    args=p.parse_args();native=Path(__file__).resolve().parent
    suffix=sysconfig.get_config_var('EXT_SUFFIX');include=sysconfig.get_paths()['include']
    if sys.platform not in ('darwin','linux') or not suffix or not include:
        raise RuntimeError('Build requires macOS/Linux and Python development headers')
    models=['crr','lr'] if args.model=='all' else [args.model]
    library=native/'libdeflate';archive=library/'build/libdeflate.a'
    if 'lr' in models:
        cmake=shutil.which('cmake') or str(Path(sys.executable).parent/'cmake')
        subprocess.run([cmake,'-S',str(library),'-B',str(library/'build'),
            '-DLIBDEFLATE_BUILD_STATIC_LIB=ON','-DLIBDEFLATE_BUILD_SHARED_LIB=OFF',
            '-DLIBDEFLATE_BUILD_GZIP=OFF','-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_POSITION_INDEPENDENT_CODE=ON'],check=True)
        subprocess.run([cmake,'--build',str(library/'build'),'--parallel',str(args.jobs)],check=True)
    for model in models:
        target=native.parent/('_'+model+'_native'+suffix)
        command=[*shlex.split(os.environ.get('CXX','c++')),'-O3','-std=c++17','-fPIC','-ffp-contract=off',
            '-I'+include,str(native/(model+'_native.cpp'))]
        if model=='lr':command+=['-I'+str(library),str(archive),'-lz']
        command+=['-bundle','-undefined','dynamic_lookup'] if sys.platform=='darwin' else ['-shared']
        subprocess.run([*command,'-o',str(target)],check=True);print(target)

if __name__=='__main__':main()
