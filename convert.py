import os
import sys
import logging
import pydicom
import argparse
import json
import SimpleITK as sitk
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, get_context

from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import RtStructInputAdapter
from dcmrtstruct2nii.exceptions import PathDoesNotExistException, ContourOutOfBoundsException

from rtstructcontour2mask import DcmPatientCoords2Mask

import logging

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='/input', help='Path to the directory containing DICOMs')
parser.add_argument('--output', default='/output', help='Path to the output directory')
parser.add_argument('--nifti', action='store_true', help='Flag to convert to nii.gz volumes')
parser.add_argument('--mhd', action='store_true', help='Flag to convert to .mhd volumes')
parser.add_argument('-l', '--label_json', default=None, help='json of labels to convert')
parser.add_argument('--anon', action='store_true', help='Flag to anonymize folders')
parser.add_argument('--mist', action='store_true', help='Flag to create mist test_path.csv')
args,_ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    handlers=[logging.StreamHandler()])

class ProgressBar:
    def __init__(self,total):
        self.count = 0
        self.next_pct = 0
        self.reset(total)

    def set_step(self):
        if self.total < 10:
            self.step = 50
        elif self.total < 100:
            self.step = 20
        else:
            self.step = 10

    def reset(self,total=None):
        if total is not None: 
            self.total = total
            self.set_step()
        self.count = 0
        self.next_pct = 0 + self.step

    def update(self,n=1):
        self.count += n
        pct = 100*self.count/self.total
        if pct >= self.next_pct:
            logging.info('{}/{} complete ({}%)'.format(self.count,self.total,int(pct)))
            self.next_pct += self.step

def load_contours(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data['labels']

def dcm_convert(dcm_paths,out_path,series_id,format='.nii.gz'):
    if not os.path.exists(os.path.normpath(out_path)): os.makedirs(os.path.normpath(out_path))
    for i,dcm_path in enumerate(dcm_paths):
        folder = os.path.basename(dcm_path)
        if i > 0:
            output_path = os.path.normpath(os.path.join(out_path,f"image_{i}_{folder}{format}"))
        else:
            output_path = os.path.normpath(os.path.join(out_path,f"image_{folder}{format}"))
        dicom_image = DcmInputAdapter().ingest(dcm_path, series_id=series_id[i])
        sitk.WriteImage(dicom_image,output_path)

def seg_convert(seg_paths,dcm_path,out_path,contours,format='.nii.gz',series_id=None):
    if not os.path.exists(os.path.normpath(out_path)): os.makedirs(os.path.normpath(out_path))
    masks = []
    dicom_image = DcmInputAdapter().ingest(dcm_path[0], series_id=series_id[0])
    size = dicom_image.GetSize()
    spacing = dicom_image.GetSpacing()
    dcm_origin = dicom_image.GetOrigin()
    for i,seg_path in enumerate(seg_paths):
        mask = np.zeros((size[2],size[0],size[1]))
        folder = os.path.basename(os.path.split(seg_path)[0])
        if i > 0:
            output_path = os.path.normpath(os.path.join(out_path, f'seg_mask_{i}_{folder}{format}'))  # make sure trailing slash is there
        else:
            output_path = os.path.normpath(os.path.join(out_path, f'seg_mask_{folder}{format}'))
        seg = pydicom.dcmread(seg_path)
        data = seg.pixel_array
        for i,slice in enumerate(seg.PerFrameFunctionalGroupsSequence):
            pos = slice.PlanePositionSequence[0].ImagePositionPatient[-1]
            pos = int(np.round(abs(dcm_origin[-1]-pos)/spacing[-1]))
            mask[pos,:,:] = data[i]
        # pos = seg.PerFrameFunctionalGroupsSequence[-1].PlanePositionSequence[0].ImagePositionPatient[-1]
        # flip = seg.PerFrameFunctionalGroupsSequence[-2].PlanePositionSequence[0].ImagePositionPatient[-1] - pos
        # seg = seg.pixel_array
        # print(seg.shape)
        # print(abs(dcm_origin[-1]-pos)/spacing[-1])
        # first_slice = int(np.round(abs(dcm_origin[-1]-pos)/spacing[-1]))
        # print(first_slice)
        # last_slice = first_slice + seg.shape[0]
        # print(last_slice)
        # print(flip)
        # if flip > 0:
        #     mask[first_slice:last_slice,:,:] = seg[::-1,:,:]
        # else:
        #     mask[first_slice:last_slice,:,:] = seg[:,:,:]
        masks.append(mask)
    try:
        mask = np.maximum.reduce(masks)
        mask = sitk.GetImageFromArray(mask)
        mask.CopyInformation(dicom_image)
        sitk.WriteImage(mask,output_path)

        logging.info('Success!')
    except Exception as ex:
        print(ex)

def mask_convert(rt_paths,dcm_path,out_path,contours,format='.nii.gz',series_id=None):
    """
    Converts DICOM RT Struct file to nii/mhd

    :param rtstruct_file: Path to the rtstruct file
    :param dicom_file: Path to the dicom file
    :param output_path: Output path where the masks are written to
    :param structures: Optional, list of structures to convert
    :param series_id: Optional, the Series Instance UID. Use  to specify the ID corresponding to the image if there are
    dicoms from more than one series in `dicom_file` folder

    :raise InvalidFileFormatException: Raised when an invalid file format is given.
    :raise PathDoesNotExistException: Raised when the given path does not exist.
    :raise UnsupportedTypeException: Raised when conversion is not supported.
    :raise ValueError: Raised when mask_background_value or mask_foreground_value is invalid.
    """

    if not os.path.exists(os.path.normpath(out_path)):
        os.makedirs(os.path.normpath(out_path))
    mask_background_value = 0
    missing = []

    for i,rt_path in enumerate(rt_paths):
        folder = os.path.basename(os.path.split(rt_path)[0])
        if i > 0:
            output_path = os.path.normpath(os.path.join(out_path, f'rs_mask_{i}_{folder}{format}'))  # make sure trailing slash is there
        else:
            output_path = os.path.normpath(os.path.join(out_path, f'rs_mask_{folder}{format}'))

        if not os.path.exists(rt_path):
            raise PathDoesNotExistException(f'rtstruct path does not exist: {rt_path}')

        if not os.path.exists(dcm_path[0]):
            raise PathDoesNotExistException(f'DICOM path does not exists: {dcm_path[0]}')

        if contours is None:
            contours = {}

        rtreader = RtStructInputAdapter()

        rtstructs = rtreader.ingest(rt_path)
        dicom_image = DcmInputAdapter().ingest(dcm_path[0], series_id=series_id[0])

        dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
        masks = []
        for i,rtstruct in enumerate(rtstructs):
            if len(contours.keys()) == 0:
                if 'sequence' not in rtstruct:
                    logging.info('Skipping mask {} no shape/polygon found'.format(rtstruct['name']))
                    continue
                
                logging.info('Working on mask {}'.format(rtstruct['name']))
                try:
                    mask = dcm_patient_coords_to_mask.convert(rtstruct['sequence'], dicom_image, mask_background_value, mask_foreground=i)
                except ContourOutOfBoundsException:
                    logging.warning(f'Structure {rtstruct["name"]} is out of bounds, ignoring contour!')
                    continue

                masks.append(mask)
            elif rtstruct['name'] in contours.keys():
                if 'sequence' not in rtstruct:
                    logging.info('Skipping mask {} no shape/polygon found'.format(rtstruct['name']))
                    continue

                logging.info('Working on mask {}'.format(rtstruct['name']))
                try:
                    mask = dcm_patient_coords_to_mask.convert(rtstruct['sequence'], dicom_image, mask_background_value,
                                                            mask_foreground=contours[rtstruct['name']])
                except ContourOutOfBoundsException:
                    logging.warning(f'Structure {rtstruct["name"]} is out of bounds, ignoring contour!')
                    continue
                
                masks.append(mask)
            else:
                missing.append(rtstruct['name'])
        try:
            mask = np.maximum.reduce(masks)
            mask = sitk.GetImageFromArray(mask)
            mask.CopyInformation(dicom_image)
            sitk.WriteImage(mask,output_path)

            logging.info('Success!')
        except Exception as ex:
            print(ex)
        return missing

def dose_convert(dose_paths,out_path,format='.nii.gz'):
    if not os.path.exists(out_path): os.makedirs(os.path.normpath(out_path))
    for dose_path in dose_paths:
        folder = os.path.basename(os.path.split(dose_path)[1])
        output_path = os.path.normpath(os.path.join(out_path,f"dose_{folder}{format}"))
        dose = pydicom.dcmread(dose_path)
        dose_out = sitk.GetImageFromArray(dose.pixel_array)
        spacing = [dose.PixelSpacing[0],dose.PixelSpacing[1],dose.SliceThickness]
        dose_out.SetSpacing(spacing)
        dose.ImageOrientationPatient.extend([0,0,1])
        dose_out.SetDirection(dose.ImageOrientationPatient)
        dose_out.SetOrigin(dose.ImagePositionPatient)
        sitk.WriteImage(dose_out, output_path)

def get_dicom_info(path):
    mod = ''
    seid = ''
    try:
        ds = pydicom.dcmread(path,specific_tags=['Modality','SeriesInstanceUID','PatientID','ReferencedSeriesSequence','FrameOfReferenceUID','ReferencedFrameOfReferenceSequence','SeriesDescription'])
        mod = ds.Modality
        seid = ds.SeriesInstanceUID
        pid = ds.PatientID
        if mod =='RTSTRUCT':
            frid = ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
            seid = ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
        elif mod == "SEG":
            frid = 0
            seid = ds.ReferencedSeriesSequence[0].SeriesInstanceUID
        else:
            #frid =ds.FrameOfReferenceUID
            frid = 0
        try:
            sed = ds.SeriesDescription.replace(" ","_").replace("/","_")
        except:
            sed = 'SeriesDes'
    except Exception as ex:
        print(ex)
        print(path)
        print(ds)
        pass
    return path,mod,seid,pid,frid,sed 

def find_dicom(input):
    
    dicom_folders = [{'dicom_folder':[]}]
    for root,_,files in os.walk(input):
        dicoms = [os.path.join(root,f) for f in files if f.endswith(".dcm")]
        if not dicoms:
            continue
        #dicom_folders[0]['dicom_folder'].extend(dicoms)
        dicom_folders.append({'dicom_folder': dicoms})

    return dicom_folders

def find_dicom_mp(dicom_folder):
    data = {}
    for dicom in dicom_folder:
            path,mod,seid,pid,frid,sed = get_dicom_info(dicom)
            if pid not in data:
                data[pid] = {}
            if seid not in data[pid]:
                data[pid][seid] = {'images':[],'mask':[],'dose':[],'seg_mask':[]}
            #if mod == 'RTDOSE' and (os.path.normpath(path), seid) not in data[pid][seid]['dose']:
            #    data[pid][seid]['dose'].append((os.path.normpath(path), seid))
            if mod == 'SEG' and (os.path.normpath(path), seid) not in data[pid][seid]['seg_mask'] and sed in ['Lung_Mets_selias','Lung_Mets_vwhite']:
                data[pid][seid]['seg_mask'].append((os.path.normpath(path), seid))
            elif mod == 'RTSTRUCT' and (os.path.normpath(path), seid) not in data[pid][seid]['mask']:
                data[pid][seid]['mask'].append((os.path.normpath(path), seid))
            elif mod in ['CT','MR','PT','MG'] and (os.path.normpath(os.path.dirname(path)), seid, sed) not in data[pid][seid]['images']:
                data[pid][seid]['images'].append((os.path.normpath(os.path.dirname(path)), seid, sed))
    return data

def get_jobs(dicoms,out_path,format,contours=None,anon=None,mist=None):
    jobs = []
    df = {'MRN':[],"Num":[]}
    df_mist = {'id':[],'image':[]}
    for data in dicoms:
        for i,pid in enumerate(data.keys()):
            for seid in data[pid].keys():
                df['MRN'].append(pid)
                df['Num'].append(i)
                if anon:
                    job = {
                    'dcm_path': None,
                    'dose_path': None,
                    'rt_path': None,
                    'image_series_id': None,
                    'mask_series_id': None,
                    'pid': pid,
                    'format': format,
                    'out_path': os.path.normpath(os.path.join(out_path,str(i),seid)),
                    'contours': contours
                    }
                    df_mist['image'].append(os.path.normpath(os.path.join(out_path,str(i),seid,'image.nii.gz')))
                else:
                    job = {
                        'dcm_path': None,
                        'dose_path': None,
                        'rt_path': None,
                        'seg_path': None,
                        'image_series_id': None,
                        'mask_series_id': None,
                        'pid': pid,
                        'format': format,
                        'out_path': os.path.normpath(os.path.join(out_path,pid,seid)),
                        'contours': contours
                    }
                    #df_mist['image'].append(os.path.normpath(os.path.join(out_path,pid,seid,'image.nii.gz')))
                if len(data[pid][seid]['images']) > 0:
                    job['dcm_path'] = [i[0] for i in data[pid][seid]['images']]
                    job['image_series_id'] = [i[1] for i in data[pid][seid]['images']]
                    job['out_path'] = os.path.normpath(os.path.join(out_path,pid,data[pid][seid]['images'][0][1]))
                    df_mist['image'].append(os.path.normpath(os.path.join(out_path,pid,data[pid][seid]['images'][0][2],'image.nii.gz')))
                    df_mist['id'].append(pid)

                if len(data[pid][seid]['mask']) > 0:
                    job['rt_path'] = [i[0] for i in data[pid][seid]['mask']]
                    job['mask_series_id'] = [i[1] for i in data[pid][seid]['mask']]
                if len(data[pid][seid]['seg_mask']) > 0:
                    job['seg_path'] = [i[0] for i in data[pid][seid]['seg_mask']]
                    job['mask_series_id'] = [i[1] for i in data[pid][seid]['seg_mask']]
                if len(data[pid][seid]['dose']) > 0:
                    job['dose_path'] = [i[0] for i in data[pid][seid]['dose']]
                jobs.append(job)
    if anon:
        if not os.path.exists(os.path.normpath(out_path)):
            os.makedirs(os.path.normpath(out_path))
        df = pd.DataFrame(df)
        df.to_excel(os.path.normpath(os.path.join(out_path,'anon_map.xlsx')))
    if mist:
        if not os.path.exists(os.path.normpath(out_path)):
            os.makedirs(os.path.normpath(out_path))
        df_mist = pd.DataFrame(df_mist)
        df_mist.to_csv(os.path.normpath(os.path.join(out_path,'test_path.csv')),index=False )
    return jobs

def convert_helper(job):
    #print(job)
    missing = []
    if job['dcm_path']:
        dcm_convert(job['dcm_path'],job['out_path'],job['image_series_id'],job['format'])
    if job['rt_path']:
        miss= mask_convert(job['rt_path'],job['dcm_path'],job['out_path'],job['contours'],job['format'],job['image_series_id'])
        missing.extend(miss)
    if job['seg_path']:
        print(job)
        seg_convert(job['seg_path'],job['dcm_path'],job['out_path'],job['contours'],job['format'],job['image_series_id'])
    # if job['dose_path']:
    #     dose_convert(job['dose_path'],job['out_path'],job['format'])
    return missing

def run_mp(items):
    err = ''
    missing = []
    try:
        miss = convert_helper(items)
        missing.extend(miss)
    except Exception as ex:
        err = 'Unable to convert {}:\n\t{}'.format(items['dcm_path'],ex)
    return err, missing

def run_find_dicom_mp(items):
    err = ''
    try:
        data = find_dicom_mp(items['dicom_folder'])
    except Exception as ex:
        data = None
        err = 'Unable to convert {}:\n\t{}'.format(items['dicom_folder'][0],ex)
    return data, err

def dicom_converter(input,out_path,format,contour_json=None,anon=None,mist=None):
    dicoms_folders = find_dicom(input)
    
    dicoms = []

    n_cpu = cpu_count() - 2
    n_cpu = min([n_cpu,len(dicoms_folders)])
    n_cpu = max([n_cpu,1])

    logging.info('Found {} folders containing DICOM. Parsing using {} threads...'.format(len(dicoms_folders),n_cpu))
    pbar = ProgressBar(len(dicoms_folders))

    err_list = []
    n_success = 0
    with get_context("spawn").Pool(processes=n_cpu) as p:
        for data,err in p.imap_unordered(run_find_dicom_mp,dicoms_folders):
            if err: 
                err_list.append(err)
            else:
                dicoms.append(data)
                n_success += 1
            pbar.update(1)
    contours = None
    if contour_json:
        contours = load_contours(contour_json)
    logging.info('Found {} contours in label json'.format(contours))

    jobs = get_jobs(dicoms,out_path,format,contours,anon,mist)

    n_cpu = cpu_count() - 2
    n_cpu = min([n_cpu,len(jobs)])
    n_cpu = max([n_cpu,1])
    logging.info('Found {} DICOM directories. Converting to {} (using {} threads)...'.format(len(jobs),format,n_cpu))

    pbar = ProgressBar(len(jobs))
    
    err_list = []
    n_success = 0
    missing_list = []
    with get_context("spawn").Pool(processes=n_cpu) as p:
        for err,miss in p.imap_unordered(run_mp,jobs):
            if err: 
                err_list.append(err)
            else:
                n_success += 1
            if miss:
                missing_list.extend(miss)
            pbar.update(1)

    logging.info('Finished! Successfully converted {}/{} {}'.format(n_success,len(jobs),format))
    if err_list:
        print('\n'.join(err_list))
        logging.warning('Found {} error(s) during conversions. Check stderr for more details.'.format(len(err_list)))
    print(missing_list)

if __name__ == "__main__":
    if args.nifti:
        format = '.nii.gz'
    elif args.mhd:
        format = '.mhd'
    dicom_converter(args.input,args.output,format,args.label_json,args.anon,args.mist)