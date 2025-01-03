[Setup]
AppName=Clustering GUI
AppVersion=1.0
WizardStyle=modern
DefaultDirName={autopf}\Clustering GUI
DefaultGroupName=Clustering GUI
OutputBaseFilename=ClusteringGUI_Setup
Compression=lzma
SolidCompression=yes
OutputDir=installer

[Files]
Source: "dist\ClusteringGUI\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\Clustering GUI"; Filename: "{app}\ClusteringGUI.exe"
Name: "{commondesktop}\Clustering GUI"; Filename: "{app}\ClusteringGUI.exe"

[Run]
Filename: "{app}\ClusteringGUI.exe"; Description: "Launch Clustering GUI"; Flags: postinstall nowait