import type { AuthResponse } from '../types';
import { apiRequest, ServiceLogger } from './api';

const categories = [
  'JuicerMixerGrinders',
  'Kettles&HotWaterDispensers',
  'HomeStorage&Organization',
  'GraphicTablets',
  'Keyboards,Mice&InputDevices',
  'Mobiles&Accessories',
  'LaptopSleeves&Slipcases',
  'Cables',
  'LaptopAccessories',
  'EggBoilers',
  'Keyboard&MouseSets',
  'RemoteControls',
  'ElectricHeaters',
  'HomeTheater,TV&Video',
  'Notebooks,WritingPads&Diaries',
  'StorageWaterHeaters',
  'Computers&Accessories',
  'SmartTelevisions',
  'LintShavers',
  'RoomHeaters',
  'PenDrives',
  'Mice',
  'USBCables',
  'Maintenance,Upkeep&Repairs',
  'Chargers',
  'Home&Kitchen',
  'Heating,Cooling&AirQuality',
  'Irons,Steamers&Accessories',
  'In-Ear',
  'InductionCooktop',
  'WaterHeaters&Geysers',
  'GeneralPurposeBatteries&BatteryChargers',
  'MixerGrinders',
  'Televisions',
  'SandwichMakers',
  'Keyboard&MiceAccessories',
  'Smartphones&BasicMobiles',
  'SmallKitchenAppliances',
  'Headphones',
  'ImmersionRods',
  'SmartWatches',
  'WearableTechnology',
  'FanHeaters',
  'Cables&Accessories',
  'PowerBanks',
  'Vacuums',
  'NetworkAdapters',
  'WallChargers',
  'WaterPurifierAccessories',
  'MousePads',
  'CeilingFans',
  'ScreenProtectors',
  'On-Ear',
  'LaundryOrganization',
  'MiniFoodProcessors&Choppers',
  'HandheldVacuums',
  'Bags&Sleeves',
  'OfficePaperProducts',
  'ExternalDevices&DataStorage',
  'VacuumSealers',
  'MobileAccessories',
  'NetworkingDevices',
  'Lapdesks',
  'Smartphones',
  'Electronics',
  'Kitchen&HomeAppliances',
  'Vacuum,Cleaning&Ironing',
  'MicroSD',
  'Accessories',
  'StylusPens',
  'Routers',
  'Vacuums&FloorCare',
  'Irons',
  'ElectricKettles',
  'DisposableBatteries',
  'Stands',
  'Headphones,Earbuds&Accessories',
  'BasicMobiles',
  'LaundryBaskets',
  'WirelessUSBAdapters',
  'Accessories&Peripherals',
  'DigitalKitchenScales',
  'Paper',
  'Kettle&ToasterSets',
  'OfficeProducts',
  'MemoryCards',
  'SteamIrons',
  'HDMICables',
  'DryIrons',
  'WaterFilters&Purifiers',
  'HandBlenders',
  'CompositionNotebooks',
  'Fans',
  'WaterPurifiers&Accessories',
  'InstantWaterHeaters',
  'Stationery',
];

export interface PreferencesRequest {
  preferences: string;
}

export const setPreferences = async (
  preferences: PreferencesRequest
): Promise<AuthResponse> => {
  ServiceLogger.logServiceCall('setPreferences', { preferences });
  return apiRequest<AuthResponse>('/api/users/preferences', 'setPreferences', {
    method: 'POST',
    body: preferences,
  });
};

export const fetchNewPreferences = async (): Promise<string> => {
  return categories.join('|');
};

export const getPreferences = async (): Promise<string> => {
  ServiceLogger.logServiceCall('getPreferences');
  return apiRequest<string>('/api/users/preferences', 'getPreferences');
};
