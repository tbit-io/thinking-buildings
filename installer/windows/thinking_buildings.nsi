; Thinking Buildings — NSIS Installer Script
; Builds a Windows MSI-style wizard installer
;
; Prerequisites:
;   1. Build with PyInstaller first:
;      PROFILE=onnx pyinstaller installer/thinking_buildings.spec
;   2. Output directory: dist/thinking-buildings-onnx/
;   3. Run: makensis installer/windows/thinking_buildings.nsi

!include "MUI2.nsh"
!include "FileFunc.nsh"

; --- Metadata ---
!define PRODUCT_NAME "Thinking Buildings"
!define PRODUCT_EXE "thinking-buildings.exe"
!define PRODUCT_PUBLISHER "TBit"
!define PRODUCT_WEB_SITE "https://tbit.io"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\${PRODUCT_EXE}"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"

; Version is injected by CI or defaults to dev
!ifndef PRODUCT_VERSION
  !define PRODUCT_VERSION "0.0.0-dev"
!endif

; Source directory — PyInstaller output
!ifndef DIST_DIR
  !define DIST_DIR "..\..\dist\thinking-buildings-onnx"
!endif

; --- Installer settings ---
Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "..\..\dist\ThinkingBuildings-${PRODUCT_VERSION}-setup.exe"
InstallDir "$PROGRAMFILES\${PRODUCT_NAME}"
InstallDirRegKey HKLM "${PRODUCT_DIR_REGKEY}" ""
ShowInstDetails show
ShowUninstDetails show
RequestExecutionLevel admin

; --- UI ---
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Welcome to ${PRODUCT_NAME} Setup"
!define MUI_WELCOMEPAGE_TEXT "This wizard will install ${PRODUCT_NAME} ${PRODUCT_VERSION} on your computer.$\r$\n$\r$\nThinking Buildings turns your camera into an intelligent security system using ML-based detection.$\r$\n$\r$\nClick Next to continue."

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; --- Install section ---
Section "Install" SecInstall
  SetOutPath "$INSTDIR"

  ; Copy all PyInstaller output
  File /r "${DIST_DIR}\*.*"

  ; Create default config in AppData if it doesn't exist
  SetOutPath "$APPDATA\ThinkingBuildings"
  IfFileExists "$APPDATA\ThinkingBuildings\config.yaml" +2
    File "..\..\config.yaml"

  ; Start Menu shortcuts
  CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\${PRODUCT_EXE}" \
    '--config "$APPDATA\ThinkingBuildings\config.yaml"' \
    "$INSTDIR\${PRODUCT_EXE}" 0
  CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall.lnk" "$INSTDIR\uninstall.exe"

  ; Registry — App Paths
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\${PRODUCT_EXE}"

  ; Registry — Add/Remove Programs
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\${PRODUCT_EXE}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"

  ; Compute installed size
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "EstimatedSize" "$0"

  ; Write uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

; --- Uninstall section ---
Section "Uninstall"
  ; Remove Start Menu
  RMDir /r "$SMPROGRAMS\${PRODUCT_NAME}"

  ; Remove install directory
  RMDir /r "$INSTDIR"

  ; Remove registry keys
  DeleteRegKey ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}"
  DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"

  ; Note: we do NOT remove %APPDATA%\ThinkingBuildings — user config stays
SectionEnd
