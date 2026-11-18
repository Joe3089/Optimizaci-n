; ─────────────────────────────────────────────────────────────────────────────
; Instalador de Windows para el Optimizador de Funciones (Inno Setup 6)
;
; Requisito previo: dist\OptimizadorFunciones.exe generado con PyInstaller.
; Se compila con scripts\build_installer.bat, o a mano:
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" installer\OptimizadorFunciones.iss
; Resultado: installer\Output\Instalador_OptimizadorFunciones_<version>.exe
;
; El instalador NO incluye dist\.env ni ninguna API Key: cada usuario configura
; la suya desde la app (se guarda en %APPDATA%\OptimizadorFunciones).
; ─────────────────────────────────────────────────────────────────────────────

#define AppName      "Optimizador de Funciones"
#define AppVersion   "1.0.0"
#define AppPublisher "Joe"
#define AppExe       "OptimizadorFunciones.exe"

[Setup]
; AppId identifica la app para actualizaciones/desinstalación: NO cambiarlo.
AppId={{6F1C2A9E-4B7D-4E3A-9C55-0D8B7A31F2E4}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\OptimizadorFunciones
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Instala sin pedir administrador (en %LOCALAPPDATA%\Programs); el usuario
; puede elegir "para todos los usuarios" en el diálogo si tiene permisos.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
SourceDir=..
OutputDir=installer\Output
OutputBaseFilename=Instalador_OptimizadorFunciones_{#AppVersion}
SetupIconFile=Menu\WindowsIcon-min.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "docs\Documentacion_Tecnica_Optimizador.pdf"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist
Source: "docs\Funciones_de_Prueba_Optimizador.pdf";   DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist
Source: "docs\ejemplos\*"; DestDir: "{app}\docs\ejemplos"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Documentación técnica"; Filename: "{app}\docs\Documentacion_Tecnica_Optimizador.pdf"
Name: "{group}\Guía de funciones de prueba"; Filename: "{app}\docs\Funciones_de_Prueba_Optimizador.pdf"
Name: "{group}\Ejemplos"; Filename: "{app}\docs\ejemplos"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
