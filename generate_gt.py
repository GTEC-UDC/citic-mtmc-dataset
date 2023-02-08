import json
import os

import pandas as pd
import numpy as np


def get_capture_range_classify(df, start: pd.Timestamp, end: pd.Timestamp,
                               timestamp_key='createdAt', classify_key='ownerEmail'):
    df = df[df[timestamp_key].between(start, end)]
    user_emails = df[classify_key].unique()
    res = {}
    for email in user_emails:
        capture_data = df[df[classify_key] == email]
        res[email] = capture_data
        # assert len(route['checkpoints.timestamp'].between(start, end)) == len(route)
    return res

class CkPtCaptureReader:
    """
    Reads the path and timesync csv files from the checkpoint app.
    """

    def __init__(self, paths_csv_file, timesync_csv_file, timesync_packet_delay=None):
        print(f'Processing {paths_csv_file} file')
        paths = pd.read_csv(paths_csv_file, parse_dates=['createdAt'])
        paths = paths.drop(columns=['updatedAt', '_id', 'routeId', 'ownerId', 'checkpoints._id', 'checkpoints.floor', '__v', 'id'])

        paths['checkpoints.timestamp'] = pd.to_datetime(paths['checkpoints.timestamp'], unit='ms', utc=True)

        last_createdAt = None
        last_ownerEmail = None
        for i, row in paths.iterrows():
            createdAt = row['createdAt']
            ownerEmail = row['ownerEmail']
            if pd.isna(createdAt):
                paths.at[i, 'createdAt'] = last_createdAt
                paths.at[i, 'ownerEmail'] = last_ownerEmail
            else:
                last_createdAt = createdAt
                last_ownerEmail = ownerEmail
                print(createdAt, ownerEmail)
        self.paths = paths

        # Tymesync processing

        timesyncs = pd.read_csv(timesync_csv_file, parse_dates=['createdAt'])
        timesyncs = timesyncs.drop(columns=['updatedAt', '_id', 'ownerId', '__v', 'id'])

        timesyncs['timestamp'] = pd.to_datetime(timesyncs['timestamp'], unit='ms', utc=True)

        timesyncs['offset'] = timesyncs['createdAt'] - timesyncs['timestamp']
        timesyncs['offset'] = timesyncs['offset'].to_numpy().astype(np.float128) / 1e9

        print()
        print('Timesyncs')

        # print(timesyncs)

        def get_est_offsets_per_user(df):
            user_emails = df['ownerEmail'].unique()
            res = {}
            for email in user_emails:
                user_data = df[df['ownerEmail'] == email]
                # res[email] = user_data['offset'].mean()
                res[email] = user_data['offset'].min()  # using min() to minimize the packet delay effects
            return res

        # print(timesyncs[ timesyncs['ownerEmail'] == 'angel@udc.es'])
        # print(timesyncs[ timesyncs['ownerEmail'] == 'valentin@udc.es'] )
        # print(timesyncs[ timesyncs['ownerEmail'] == 'invitado1@udc.es'])

        mean_offset_per_user = get_est_offsets_per_user(timesyncs)
        # get the minimum absolute offset of all users
        min_abs_offset = min(abs(mean_offset_per_user[k]) for k in mean_offset_per_user)
        if timesync_packet_delay is None:
            timesync_packet_delay = min_abs_offset

        # We assume that one user is perfectly synchronized with the server and his/her offset is produced by packet delay. We remove this delay for all the users.

        print('Assuming a packet delay of', timesync_packet_delay, 'seconds.')
        for email, offset in mean_offset_per_user.items():
            mean_offset_per_user[email] = round((offset - timesync_packet_delay) * 1000) / 1000.0

        print('Mean offsets per user:', mean_offset_per_user)

        print('Correcting path information')
        for i, row in self.paths.iterrows():
            email = row['ownerEmail']
            if email in mean_offset_per_user:
                offset = mean_offset_per_user[email]
                self.paths.at[i, 'checkpoints.timestamp'] = self.paths.at[i, 'checkpoints.timestamp'] + pd.Timedelta(seconds=offset)

    def get_capture(self, start: pd.Timestamp, end: pd.Timestamp):
        return get_capture_range_classify(self.paths, start, end, 'createdAt', 'ownerEmail')


def get_position_at_time(track, time):
    """
    Returns the position of the track at the given time using linear interpolation.

    :param track: must be a NxM numpy array, sorted by the first column (time): [time, x, y,...].
    :param time: time index
    :return: np.array of shape (M-1,) with the interpolated coordinates
    """
    if time < track[0, 0] or time > track[-1, 0]:
        return None
    ind = np.searchsorted(track[:, 0], time)
    if ind == 0:
        return track[0, 1:]
    t1 = track[ind - 1, 0]
    t2 = track[ind, 0]
    pos1 = track[ind - 1, 1:]
    pos2 = track[ind, 1:]
    return pos1 + (pos2 - pos1) * ((time - t1) / (t2 - t1))


class CkPtPathInterpolator:
    """
    Class that parses the CkPtCaptureReader output paths and provides the position of each person at any timestamp using
    linear interpolation.
    """

    def __init__(self, capture_dict, ref_time: pd.Timestamp):
        """"
        :param capture: CkPtCaptureReader.get_capture() output
        """
        self.capture_dict = capture_dict
        self.tracks = {user: df[['checkpoints.timestamp', 'checkpoints.x', 'checkpoints.y']].values
                       for user, df in capture_dict.items()}
        if ref_time is not None:
            self.ref_time = ref_time
        else:
            self.ref_time = min((df['checkpoints.timestamp'].min() for df in self.capture_dict.values()))

    def get_user_position_at_time(self, user, time=None, offset=0.0):
        """
        Returns the position of the track at the given time using linear interpolation.

        :param user: user id
        :param time: timestamp in pd.Timestamp or other compatible format. If None, ref_time is used.
        :param offset: offset in seconds
        :return: np.array of shape (2,) with (x, y) coordinates
        """
        assert user in self.tracks, 'User {} not found in the capture'.format(user)
        if time is None:
            time = self.ref_time
        if not isinstance(time, pd.Timestamp):
            time = pd.Timestamp(time)
        time = time + pd.Timedelta(seconds=offset)
        track = self.tracks[user]
        return get_position_at_time(track, time)

    def get_positions_at_time(self, time=None, offset=0.0):
        """
        Returns the positions of the users at the given time using linear interpolation.

        :param time: timestamp in pd.Timestamp or other compatible format. If None, ref_time is used.
        :param offset: offset in seconds
        :return: dict of user id to np.array of shape (2,) with (x, y) coordinates
        """
        if time is None:
            time = self.ref_time
        if not isinstance(time, pd.Timestamp):
            time = pd.Timestamp(time)
        time = time + pd.Timedelta(seconds=offset)

        return {user: self.get_user_position_at_time(user, time) for user in self.tracks}

    def write_to_mot_file(self, filename, fps, start_time, end_time, overwrite=False):
        """
        Writes the interpolated positions to a file in MOT format:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        The bounding box values are set to -1. The z value is set to 0.
        The start time is set to the ref_time + start_offset.

        :param filename: output filename. If already exists, it will be overwritten.
        :param fps: frames per second
        :param start_time: start time in pd.Timestamp or other compatible format.
        :param end_time: end time in pd.Timestamp or other compatible format.
        """

        # Check if the file already exists
        if os.path.exists(filename) and not overwrite:
            print(f'File {filename} already exists. Use overwrite=True to overwrite it.')
            return

        user_to_id = {user: i+1 for i, user in enumerate(self.tracks.keys())}
        time_incr = pd.Timedelta(seconds=1 / fps)
        time = start_time

        with open(filename, 'w') as f:
            frame_num = 1
            while time < end_time:
                positions = self.get_positions_at_time(time)
                for user, pos in positions.items():
                    if pos is not None:
                        f.write('{},{},-1,-1,-1,-1,-1,{},{},0\n'.format(frame_num, user_to_id[user], *pos))
                time += time_incr
                frame_num += 1


class BeaconReportReader:
    """
    Reads the beacon reports from the checkpoint app in JSON format.
    """

    def __init__(self, beacon_report_file):
        print(f'Reading {beacon_report_file} file')
        with open(beacon_report_file) as file:
            beacon_reports = json.load(file)

        beacon_reports = pd.DataFrame(beacon_reports['data'])
        beacon_reports = beacon_reports.drop(columns=['updatedAt', '_id', 'ownerId', '__v'])
        beacon_reports['createdAt'] = pd.to_datetime(beacon_reports['createdAt'])
        self.beacon_reports = beacon_reports

    def get_beacon_capture(self, capture_start, capture_end):
        # Returns a dictionary with the beacon reports for the given capture range.
        # The dictionary is indexed by the ownerEmail. The value is a DataFrame with the beacon reports for that user.
        beacon_reports = self.beacon_reports
        beacons_dict = get_capture_range_classify(beacon_reports, capture_start, capture_end, 'createdAt',
                                                  'ownerEmail')
        scans = {}
        for email, data in beacons_dict.items():
            scans[email] = pd.json_normalize(data.to_dict('records'), 'scans', ['createdAt']).drop(columns=['_id'])
            print(
                f'{email} has {len(data)} beacon reports in range {data["createdAt"].min()} - {data["createdAt"].max()} with {len(scans[email])} scans.')
            # print(scans[email]['rssi'].describe())
        return scans


def write_BLE_measurements(file_name, beacon_reports, overwrite=False, timestamp_offset=0.0):
    """
    Writes the BLE measurements to a JSON file with the format expected by the rssproximitytomqtt application.
    :param file_name: the file name to write to.
    :param beacon_reports: the beacon reports dictionary to write, as it was returned by BeaconReportReader.get_beacon_capture().
    :param overwrite: if True, overwrites the file if it already exists.
    """

    # Check if the file already exists
    if os.path.exists(file_name) and not overwrite:
        print(f'File {file_name} already exists. Use overwrite=True to overwrite it.')
        return

    # Merge beacon_reports
    df_list = []
    for email, dataframe in beacon_reports.items():
        df = dataframe.copy()
        df['email'] = email
        df_list.append(df)
    merged_br = pd.concat(df_list, ignore_index=True)
    merged_br.sort_values(by='createdAt', inplace=True)

    with open(file_name, mode='w') as out_file:
        for ind, row in merged_br.iterrows():
            # anchor_id = str(row['major']) + ':' + str(row['minor'])
            # Sevenix beacons are identified by the major value. minor value informs about the battery level
            anchor_id = str(row['major'])
            timestamp = row['createdAt']
            user_id = row['email']
            rssi = row['rssi']
            if rssi == 0.0:  # Fix a bug in iOS application: ignore beacons with 0.0 rssi.
                continue

            def beacon_report_to_dict(anchor_id, timestamp, user_id, rssi):
                timestr = timestamp.replace(tzinfo=None).isoformat()
                return {
                    "NodeID": anchor_id,
                    "Timestamp": timestr,
                    "Beacons": [
                        {"Timestamp": timestr,
                         "Addr": user_id,
                         "RSSI": float(rssi),
                         "UUID": "", "TerminalID": "", "RoomID": "", "Location": ""}
                    ]}

            out_dict = beacon_report_to_dict(anchor_id, timestamp+pd.Timedelta(timestamp_offset, unit='sec'), user_id, rssi)
            json.dump(out_dict, out_file)
            out_file.write('\n')



if __name__ == '__main__':
    data_dir = './'
    ble_dir = 'BLE/'

    capture_reader = CkPtCaptureReader(data_dir + 'paths.csv', data_dir + 'timesyncs.csv')

    overwrite = False

    print()
    print('Getting CITIC captures of 2021-10-06')
    print('Capture 1')
    capture1_start = pd.Timestamp('2021-10-06 20:07:00+00:00')
    capture1_end = pd.Timestamp('2021-10-06 20:11:00+00:00')
    capture1_dict = capture_reader.get_capture(capture1_start, capture1_end)
    for email, route in capture1_dict.items():
        print(email, 'performed route', route.iloc[0]['routeTag'], 'with', len(route), 'checkpoints')

    print('Capture 2')
    capture2_start = pd.Timestamp('2021-10-06 20:13:30+00:00')
    capture2_end = pd.Timestamp('2021-10-06 20:18:30+00:00')
    capture2_dict = capture_reader.get_capture(capture2_start, capture2_end)
    for email, route in capture2_dict.items():
        print(email, 'performed route', route.iloc[0]['routeTag'], 'with', len(route), 'checkpoints')

    print('Capture 3')
    capture3_start = pd.Timestamp('2021-10-06 20:29:50+00:00')
    capture3_end = pd.Timestamp('2021-10-06 20:33:00+00:00')
    capture3_dict = capture_reader.get_capture(capture3_start, capture3_end)
    for email, route in capture3_dict.items():
        print(email, 'performed route', route.iloc[0]['routeTag'], 'with', len(route), 'checkpoints')

    print()
    print(f"Writing the GT files of the captures in {data_dir} directory.")

    capture1_start_video = pd.Timestamp('2021-10-06 20:07:51+00:00')
    capture1_end_video = pd.Timestamp('2021-10-06 20:10:43.500+00:00')
    # copy capture1_dict without invitado1@udc.es user
    capture1_dict_no_invitado = {key: value for key, value in capture1_dict.items() if key != 'invitado1@udc.es'}
    ckpt_interpolator = CkPtPathInterpolator(capture1_dict_no_invitado, capture1_start_video)
    ckpt_interpolator.write_to_mot_file(data_dir + 'capture1_gt.mot.txt', 25, capture1_start_video, capture1_end_video,
                                        overwrite=overwrite)
    capture2_start_video = pd.Timestamp('2021-10-06 20:14:46+00:00')
    capture2_end_video = pd.Timestamp('2021-10-06 20:18:03.500+00:00')
    capture2_dict_no_invitado = {key: value for key, value in capture2_dict.items() if key != 'invitado1@udc.es'}
    ckpt_interpolator = CkPtPathInterpolator(capture2_dict_no_invitado, capture2_start_video)
    ckpt_interpolator.write_to_mot_file(data_dir + 'capture2_gt.mot.txt', 25, capture2_start_video, capture2_end_video,
                                        overwrite=overwrite)
    capture3_start_video = pd.Timestamp('2021-10-06 20:29:58.560+00:00')
    capture3_end_video = pd.Timestamp('2021-10-06 20:32:42.060+00:00')
    capture3_dict_no_invitado = {key: value for key, value in capture3_dict.items() if key != 'invitado1@udc.es'}
    ckpt_interpolator = CkPtPathInterpolator(capture3_dict_no_invitado, capture3_start_video)
    ckpt_interpolator.write_to_mot_file(data_dir + 'capture3_gt.mot.txt', 25, capture3_start_video, capture3_end_video,
                                        overwrite=overwrite)


    # Beacon reports
    print()
    print('Beacon reports')

    brr = BeaconReportReader(ble_dir + 'beaconreports.json')

    ble_capture_ranges = {
        'capture1': (capture1_start_video, capture1_end_video),
        'capture2': (capture2_start_video, capture2_end_video),
        'capture3': (capture3_start_video, capture3_end_video)
    }
    print('Writing BLE measurements')
    for capture_name, (capture_start, capture_end) in ble_capture_ranges.items():
        print()
        print(f'{capture_name} : {capture_start} - {capture_end}')
        beacon_scans = brr.get_beacon_capture(capture_start, capture_end)

        fname = ble_dir + capture_name + '_beacon_scans.json'
        if os.path.exists(fname) and not overwrite:
            print(fname + ' already exists. Skipping.')
            continue
        print(f'Writing {fname}')
        # Write the beacon scans relative to the unix epoch (1970-01-01T00:00:00.00)
        write_BLE_measurements(fname, beacon_scans, overwrite=overwrite, timestamp_offset=-capture_start.timestamp())

